# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from argparse import ArgumentParser
from time import perf_counter
from typing import List, Optional, Tuple

import editdistance
import pynini
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.data_loader_utils import post_process_punct, pre_process
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.utils_audio_based import get_alignment
from nemo_text_processing.utils.logging import logger

"""
The script provides multiple normalization options and chooses the best one that minimizes CER of the ASR output
(most of the semiotic classes use deterministic=False flag).

To run this script with a .json manifest file, the manifest file should contain the following fields:
    "text" - raw text (could be changed using "--manifest_text_field")
    "pred_text" - ASR model prediction, see https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py 
        on how to transcribe manifest
    
    Example for a manifest line:
        {"text": "In December 2015, ...", "pred_txt": "on december two thousand fifteen"}

    When the manifest is ready, run:
        python normalize_with_audio.py \
            --manifest PATH/TO/MANIFEST.JSON \
            --language en \
            --output_filename=<PATH TO OUTPUT .JSON MANIFEST> \
            --n_jobs=-1 \
            --batch_size=300 \
            --manifest_text_field="text"

To see possible normalization options for a text input without an audio file (could be used for debugging), run:
    python python normalize_with_audio.py --text "RAW TEXT"

Specify `--cache_dir` to generate .far grammars once and re-used them for faster inference
"""


class NormalizerWithAudio(Normalizer):
    """
    Normalizer class that converts text from written to spoken form.
    Useful for TTS preprocessing.

    Args:
        input_case: expected input capitalization
        lang: language
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        post_process: WFST-based post processing, e.g. to remove extra spaces added during TN.
            Note: punct_post_process flag in normalize() supports all languages.
        max_number_of_permutations_per_split: a maximum number
                of permutations which can be generated from input sequence of tokens.
    """

    def __init__(
        self,
        input_case: str,
        lang: str = 'en',
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        lm: bool = False,
        post_process: bool = True,
        max_number_of_permutations_per_split: int = 729,
    ):

        # initialize non-deterministic normalizer
        super().__init__(
            input_case=input_case,
            lang=lang,
            deterministic=False,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
            lm=lm,
            post_process=post_process,
        )
        self.tagger_non_deterministic = self.tagger
        self.verbalizer_non_deterministic = self.verbalizer

        if lang != "ru":
            # initialize deterministic normalizer
            super().__init__(
                input_case=input_case,
                lang=lang,
                deterministic=True,
                cache_dir=cache_dir,
                overwrite_cache=overwrite_cache,
                whitelist=whitelist,
                lm=lm,
                post_process=post_process,
                max_number_of_permutations_per_split=max_number_of_permutations_per_split,
            )
        else:
            self.tagger, self.verbalizer = None, None
        self.lm = lm

    def normalize(
        self,
        text: str,
        n_tagged: int,
        punct_post_process: bool = True,
        verbose: bool = False,
        pred_text: Optional[str] = None,
        cer_threshold: float = -1,
        **kwargs,
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            n_tagged: number of tagged options to consider, -1 - to get all possible tagged options
            punct_post_process: whether to normalize punctuation
            verbose: whether to print intermediate meta information
            pred_text: ASR model transcript
            cer_threshold: if CER for pred_text and the normalization option is above the cer_threshold,
                default deterministic normalization will be used. Set to -1 to disable cer-based filtering.
                Specify the value in %, e.g. 100 not 1.

        Returns:
            normalized text options (usually there are multiple ways of normalizing a given semiotic class)
        """
        if pred_text is None or pred_text == "" or self.tagger is None:
            return self.normalize_non_deterministic(
                text=text, n_tagged=n_tagged, punct_post_process=punct_post_process, verbose=verbose
            )

        try:
            det_norm = super().normalize(
                text=text, verbose=verbose, punct_pre_process=False, punct_post_process=punct_post_process
            )
        except RecursionError:
            raise RecursionError(f"RecursionError. Try decreasing --max_number_of_permutations_per_split")

        semiotic_spans, pred_text_spans, norm_spans, text_with_span_tags_list, masked_idx_list = get_alignment(
            text, det_norm, pred_text, verbose=False
        )

        sem_tag_idx = 0
        for cur_semiotic_span, cur_pred_text, cur_deter_norm in zip(semiotic_spans, pred_text_spans, norm_spans):
            if len(cur_semiotic_span) == 0:
                text_with_span_tags_list[masked_idx_list[sem_tag_idx]] = ""
            else:
                non_deter_options = self.normalize_non_deterministic(
                    text=cur_semiotic_span, n_tagged=n_tagged, punct_post_process=punct_post_process, verbose=verbose,
                )
                try:
                    best_option, cer, _ = self.select_best_match(
                        normalized_texts=non_deter_options, pred_text=cur_pred_text, verbose=verbose,
                    )
                    if cer_threshold > 0 and cer > cer_threshold:
                        best_option = cur_deter_norm
                        if verbose:
                            logger.info(
                                f"CER of the best normalization option is above cer_theshold, using determinictis option. CER: {cer}"
                            )
                except:
                    # fall back to the default normalization option
                    best_option = cur_deter_norm

                text_with_span_tags_list[masked_idx_list[sem_tag_idx]] = best_option
            sem_tag_idx += 1

        normalized_text = " ".join(text_with_span_tags_list)
        return normalized_text.replace("  ", " ")

    def normalize_non_deterministic(
        self, text: str, n_tagged: int, punct_post_process: bool = True, verbose: bool = False
    ):
        # get deterministic option
        if self.tagger:
            deterministic_form = super().normalize(
                text=text, verbose=verbose, punct_pre_process=False, punct_post_process=punct_post_process
            )
        else:
            deterministic_form = None

        original_text = text

        text = pre_process(text)  # to handle []
        text = text.strip()
        if not text:
            if verbose:
                logger.info(text)
            return text

        text = pynini.escape(text)
        if self.lm:
            if self.lang not in ["en"]:
                raise ValueError(f"{self.lang} is not supported in LM mode")

            if self.lang == "en":
                # this to keep arpabet phonemes in the list of options
                if "[" in text and "]" in text:

                    lattice = rewrite.rewrite_lattice(text, self.tagger_non_deterministic.fst)
                else:
                    try:
                        lattice = rewrite.rewrite_lattice(text, self.tagger_non_deterministic.fst_no_digits)
                    except pynini.lib.rewrite.Error:
                        lattice = rewrite.rewrite_lattice(text, self.tagger_non_deterministic.fst)
                lattice = rewrite.lattice_to_nshortest(lattice, n_tagged)
                tagged_texts = [(x[1], float(x[2])) for x in lattice.paths().items()]
                tagged_texts.sort(key=lambda x: x[1])
                tagged_texts, weights = list(zip(*tagged_texts))
        else:
            tagged_texts = self._get_tagged_text(text, n_tagged)

        # non-deterministic Eng normalization uses tagger composed with verbalizer, no permutation in between
        if self.lang == "en":
            normalized_texts = tagged_texts
            normalized_texts = [self.post_process(text) for text in normalized_texts]
        else:
            normalized_texts = []
            for tagged_text in tagged_texts:
                self._verbalize(tagged_text, normalized_texts, n_tagged, verbose=verbose)

        if len(normalized_texts) == 0:
            logger.warning("Failed text: " + text + ", normalized_texts: " + str(normalized_texts))
            return text

        if punct_post_process:
            # do post-processing based on Moses detokenizer
            if self.moses_detokenizer:
                normalized_texts = [self.moses_detokenizer.detokenize([t]) for t in normalized_texts]
                normalized_texts = [
                    post_process_punct(input=original_text, normalized_text=t) for t in normalized_texts
                ]

        if self.lm:
            remove_dup = sorted(list(set(zip(normalized_texts, weights))), key=lambda x: x[1])
            normalized_texts, weights = zip(*remove_dup)
            return list(normalized_texts), weights

        if deterministic_form is not None:
            normalized_texts.append(deterministic_form)

        normalized_texts = set(normalized_texts)
        return normalized_texts

    def normalize_line(
        self,
        n_tagged: int,
        line: str,
        verbose: bool = False,
        punct_pre_process=False,
        punct_post_process=True,
        text_field: str = "text",
        asr_pred_field: str = "pred_text",
        output_field: str = "normalized",
        cer_threshold: float = -1,
    ):
        """
        Normalizes "text_field" in line from a .json manifest

        Args:
            n_tagged: number of normalization options to return
            line: line of a .json manifest
            verbose: set to True to see intermediate output of normalization
            punct_pre_process: set to True to do punctuation pre-processing
            punct_post_process: set to True to do punctuation post-processing
            text_field: name of the field in the manifest to normalize
            asr_pred_field: name of the field in the manifest with ASR predictions
            output_field: name of the field in the manifest to save normalized text
            cer_threshold: if CER for pred_text and the normalization option is above the cer_threshold,
                default deterministic normalization will be used. Set to -1 to disable cer-based filtering.
                Specify the value in %, e.g. 100 not 1.
        """
        line = json.loads(line)

        normalized_text = self.normalize(
            text=line[text_field],
            verbose=verbose,
            n_tagged=n_tagged,
            punct_post_process=punct_post_process,
            pred_text=line[asr_pred_field],
            cer_threshold=cer_threshold,
        )
        line[output_field] = normalized_text
        return line

    def _get_tagged_text(self, text, n_tagged):
        """
        Returns text after tokenize and classify
        Args;
            text: input  text
            n_tagged: number of tagged options to consider, -1 - return all possible tagged options
        """
        if n_tagged == -1:
            if self.lang == "en":
                # this to keep arpabet phonemes in the list of options
                if "[" in text and "]" in text:
                    tagged_texts = rewrite.rewrites(text, self.tagger_non_deterministic.fst)
                else:
                    try:
                        tagged_texts = rewrite.rewrites(text, self.tagger_non_deterministic.fst_no_digits)
                    except pynini.lib.rewrite.Error:
                        tagged_texts = rewrite.rewrites(text, self.tagger_non_deterministic.fst)
            else:
                tagged_texts = rewrite.rewrites(text, self.tagger_non_deterministic.fst)
        else:
            if self.lang == "en":
                # this to keep arpabet phonemes in the list of options
                if "[" in text and "]" in text:
                    tagged_texts = rewrite.top_rewrites(text, self.tagger_non_deterministic.fst, nshortest=n_tagged)
                else:
                    try:
                        # try self.tagger graph that produces output without digits
                        tagged_texts = rewrite.top_rewrites(
                            text, self.tagger_non_deterministic.fst_no_digits, nshortest=n_tagged
                        )
                    except pynini.lib.rewrite.Error:
                        tagged_texts = rewrite.top_rewrites(
                            text, self.tagger_non_deterministic.fst, nshortest=n_tagged
                        )
            else:
                tagged_texts = rewrite.top_rewrites(text, self.tagger_non_deterministic.fst, nshortest=n_tagged)
        return tagged_texts

    def _verbalize(self, tagged_text: str, normalized_texts: List[str], n_tagged: int, verbose: bool = False):
        """
        Verbalizes tagged text

        Args:
            tagged_text: text with tags
            normalized_texts: list of possible normalization options
            verbose: if true prints intermediate classification results
        """

        def get_verbalized_text(tagged_text):
            return rewrite.top_rewrites(tagged_text, self.verbalizer_non_deterministic.fst, n_tagged)

        self.parser(tagged_text)
        tokens = self.parser.parse()
        tags_reordered = self.generate_permutations(tokens)
        for tagged_text_reordered in tags_reordered:
            try:
                tagged_text_reordered = pynini.escape(tagged_text_reordered)
                normalized_texts.extend(get_verbalized_text(tagged_text_reordered))
                if verbose:
                    logger.info(tagged_text_reordered)

            except pynini.lib.rewrite.Error:
                continue

    def select_best_match(
        self, normalized_texts: List[str], pred_text: str, verbose: bool = False, remove_punct: bool = False,
    ):
        """
        Selects the best normalization option based on the lowest CER

        Args:
            normalized_texts: normalized text options
            pred_text: ASR model transcript of the audio file corresponding to the normalized text
            verbose: whether to print intermediate meta information
            remove_punct: whether to remove punctuation before calculating CER

        Returns:
            normalized text with the lowest CER and CER value
        """
        normalized_texts_cer = calculate_cer(normalized_texts, pred_text, remove_punct)
        normalized_texts_cer = sorted(normalized_texts_cer, key=lambda x: x[1])
        normalized_text, cer, idx = normalized_texts_cer[0]

        if verbose:
            logger.info('-' * 30)
            for option in normalized_texts:
                logger.info(option)
            logger.info('-' * 30)
        return normalized_text, cer, idx


def calculate_cer(normalized_texts: List[str], pred_text: str, remove_punct=False) -> List[Tuple[str, float]]:
    """
    Calculates character error rate (CER)

    Args:
        normalized_texts: normalized text options
        pred_text: ASR model output

    Returns: normalized options with corresponding CER
    """
    normalized_options = []
    for i, text in enumerate(normalized_texts):
        text_clean = text.replace('-', ' ').lower()
        if remove_punct:
            for punct in "!?:;,.-()*+-/<=>@^_":
                text_clean = text_clean.replace(punct, " ").replace("  ", " ")

        cer = editdistance.eval(pred_text, text_clean) * 100.0 / len(pred_text)
        normalized_options.append((text, cer, i))
    return normalized_options


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--text", help="input string or path to a .txt file", default=None, type=str)
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument(
        "--language", help="Select target language", choices=["en", "ru", "de", "es", "sv"], default="en", type=str
    )
    parser.add_argument("--manifest", default=None, help="path to .json manifest")
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Path of where to save .json manifest with normalization outputs."
        " It will only be saved if --manifest is a .json manifest.",
        type=str,
    )
    parser.add_argument(
        '--manifest_text_field',
        help="A field in .json manifest to normalize (applicable only --manifest is specified)",
        type=str,
        default="text",
    )
    parser.add_argument(
        '--manifest_asr_pred_field',
        help="A field in .json manifest with ASR predictions (applicable only --manifest is specified)",
        type=str,
        default="pred_text",
    )
    parser.add_argument(
        "--n_tagged",
        type=int,
        default=30,
        help="number of tagged options to consider, -1 - return all possible tagged options",
    )
    parser.add_argument("--verbose", help="print info for debugging", action="store_true")
    parser.add_argument(
        "--no_remove_punct_for_cer",
        help="Set to True to NOT remove punctuation before calculating CER",
        action="store_true",
    )
    parser.add_argument(
        "--no_punct_post_process", help="set to True to disable punctuation post processing", action="store_true"
    )
    parser.add_argument("--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true")
    parser.add_argument("--whitelist", help="path to a file with with whitelist", default=None, type=str)
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
    parser.add_argument(
        "--lm", action="store_true", help="Set to True for WFST+LM. Only available for English right now."
    )
    parser.add_argument(
        "--cer_threshold",
        default=-1,
        type=float,
        help="if CER for pred_text and the normalization option is above the cer_threshold, default deterministic normalization will be used. Set to -1 to disable cer-based filtering. Specify the value in %, e.g. 100 not 1.",
    )
    parser.add_argument("--batch_size", default=200, type=int, help="Number of examples for each process")
    parser.add_argument(
        "--max_number_of_permutations_per_split",
        default=729,
        type=int,
        help="a maximum number of permutations which can be generated from input sequence of tokens.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.whitelist = os.path.abspath(args.whitelist) if args.whitelist else None
    if args.text is not None:
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
            lm=args.lm,
            max_number_of_permutations_per_split=args.max_number_of_permutations_per_split,
        )
        start = perf_counter()
        if os.path.exists(args.text):
            with open(args.text, 'r') as f:
                args.text = f.read().strip()

        options = normalizer.normalize(
            text=args.text,
            n_tagged=args.n_tagged,
            punct_post_process=not args.no_punct_post_process,
            verbose=args.verbose,
        )
        for option in options:
            logger.info(option)
    elif args.manifest.endswith('.json'):
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
            max_number_of_permutations_per_split=args.max_number_of_permutations_per_split,
        )
        start = perf_counter()
        normalizer.normalize_manifest(
            manifest=args.manifest,
            n_jobs=args.n_jobs,
            punct_pre_process=True,
            punct_post_process=not args.no_punct_post_process,
            batch_size=args.batch_size,
            output_filename=args.output_filename,
            n_tagged=args.n_tagged,
            text_field=args.manifest_text_field,
            asr_pred_field=args.manifest_asr_pred_field,
            cer_threshold=args.cer_threshold,
            verbose=args.verbose,
        )
    else:
        raise ValueError(
            "Provide either path to .json manifest with '--manifest' OR "
            + "an input text with '--text' (for debugging without audio)"
        )
    logger.info(f'Execution time: {round((perf_counter() - start)/60, 2)} min.')
