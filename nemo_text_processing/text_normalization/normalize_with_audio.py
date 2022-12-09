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
import shutil
from argparse import ArgumentParser
from glob import glob
from time import perf_counter
from typing import List, Optional, Tuple

import Levenshtein
import pynini
from joblib import Parallel, delayed
from nemo_text_processing.fst_alignment.alignment import (
    create_symbol_table,
    get_string_alignment,
    get_word_segments,
    indexed_map_to_output,
)
from nemo_text_processing.text_normalization.data_loader_utils import post_process_punct, pre_process
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.utils_audio_based import SEMIOTIC_TAG, get_alignment, get_semiotic_spans
from pynini import Far
from pynini.lib import rewrite
from tqdm import tqdm

try:
    # from nemo.collections.asr.metrics.wer import word_error_rate

    ASR_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    ASR_AVAILABLE = False


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
        )

        fst_tc = f"{cache_dir}/en_tn_True_deterministic_cased__tokenize.far"
        fst_ver = f"{cache_dir}/en_tn_True_deterministic_verbalizer.far"
        fst_punct_post = f"{cache_dir}/en_tn_post_processing.far"
        fst_tc = Far(fst_tc, mode='r')['tokenize_and_classify']
        fst_ver = Far(fst_ver, mode='r')['verbalize']
        fst_punct_post = Far(fst_punct_post, mode='r')['post_process_graph']
        self.merged_tn_deterministic_graph = (fst_tc @ fst_ver) @ fst_punct_post
        self.symbol_table = create_symbol_table()
        self.lm = lm

    def _process_semiotic_span(
        self, semiotic_span, pred_text, n_tagged, punct_post_process, verbose, cer_threshold=100
    ):
        try:
            options = self.normalize_non_deterministic(
                text=semiotic_span, n_tagged=n_tagged, punct_post_process=punct_post_process, verbose=verbose
            )
        except:
            # TODO: fall back to the default normalization -> restore from the alignment
            options = ["DEFAULT"]

        print("=" * 40)
        print(semiotic_span)
        [print(x) for x in options]

        best_option, cer = self.select_best_match(
            normalized_texts=options,
            input_text=semiotic_span,
            pred_text=pred_text,
            verbose=verbose,
            cer_threshold=cer_threshold,
        )
        return best_option

    # def normalize_split_text_approach(
    #     self,
    #     text: str,
    #     n_tagged: int,
    #     punct_post_process: bool = True,
    #     verbose: bool = False,
    #     pred_text: str = None,
    #     **kwargs,
    # ) -> str:
    #     """
    #     Main function. Normalizes tokens from written to spoken form
    #         e.g. 12 kg -> twelve kilograms
    #
    #     Args:
    #         text: string that may include semiotic classes
    #         n_tagged: number of tagged options to consider, -1 - to get all possible tagged options
    #         punct_post_process: whether to normalize punctuation
    #         verbose: whether to print intermediate meta information
    #
    #     Returns:
    #         normalized text options (usually there are multiple ways of normalizing a given semiotic class)
    #     """
    #
    #     text_list = [text]
    #     if len(text.split()) > 500:
    #         print(
    #             "Your input is too long. Please split up the input into sentences, "
    #             "or strings with fewer than 500 words"
    #         )
    #         print("split by sentences")
    #         text_list = self.split_text_into_sentences(text)
    #
    #     semiotic_spans = []
    #     semiotic_spans_det_norm = []
    #     masked_idx_list = []
    #
    #     for i in range(len(text_list)):
    #         t = text_list[i]
    #         cur_det_norm = super().normalize(
    #             text=t, verbose=verbose, punct_pre_process=False, punct_post_process=punct_post_process
    #         )
    #         if t != cur_det_norm:
    #             _, cur_sem_span, text_list[i], cur_masked_idx, cur_det_norm_ = get_semiotic_spans(t, cur_det_norm)
    #             masked_idx_list.append(cur_masked_idx)
    #             semiotic_spans_det_norm.append(cur_det_norm_)
    #             semiotic_spans.append(cur_sem_span)
    #         else:
    #             # TODO refactor to avoid this
    #             text_list[i] = t.split()
    #             masked_idx_list.append([])
    #             semiotic_spans.append([])
    #             semiotic_spans_det_norm.append([])
    #     try:
    #         # no semiotic spans
    #         if sum([len(x) for x in semiotic_spans]) == 0:
    #             normalized_text = ""
    #             for sent in text_list:
    #                 normalized_text += " ".join(sent)
    #             return normalized_text
    #     except Exception as e:
    #         print(sent)
    #         print(normalized_text)
    #         print(f"ERROR: {e}")
    #         import pdb
    #
    #         pdb.set_trace()
    #         print()
    #     try:
    #         # replace all but the target with det_norm option
    #         for sent_idx, cur_masked_idx_list in enumerate(masked_idx_list):
    #             for i, semiotic_idx in enumerate(cur_masked_idx_list):
    #                 text_list[sent_idx][semiotic_idx] = semiotic_spans_det_norm[sent_idx][i]
    #     except:
    #         import pdb
    #
    #         pdb.set_trace()
    #         print()
    #
    #     # create texts to compare against pred_text, all but the current semiotic span use default normalization option # TODO - use the best for processed spans?
    #     texts_for_cer = []
    #     audio_based_options = []
    #     for sent_idx, cur_masked_idx_list in enumerate(masked_idx_list):
    #         texts_for_cer_sent = []
    #         audio_based_options_sent = []
    #         for i in range(len(cur_masked_idx_list)):
    #             try:
    #                 options = self.normalize_non_deterministic(
    #                     text=semiotic_spans[sent_idx][i],
    #                     n_tagged=n_tagged,
    #                     punct_post_process=punct_post_process,
    #                     verbose=verbose,
    #                 )
    #             except:
    #                 options = semiotic_spans_det_norm[sent_idx][i]
    #
    #             # replace default normalization options for the current span with each possible audio-based option
    #             cur_texts_for_cer = []
    #             cur_audio_based_options = []
    #             for option in options:
    #                 cur_audio_based_options.append(option)
    #                 semiotic_idx = cur_masked_idx_list[i]
    #                 cur_text = text_list[sent_idx][:semiotic_idx] + [option] + text_list[sent_idx][semiotic_idx + 1 :]
    #                 cur_texts_for_cer.append(" ".join(cur_text))
    #             texts_for_cer_sent.append(cur_texts_for_cer)
    #             audio_based_options_sent.append(cur_audio_based_options)
    #         texts_for_cer.append(texts_for_cer_sent)
    #         audio_based_options.append(audio_based_options_sent)
    #
    #     selected_options = []
    #     for sent_idx, cur_sent in enumerate(texts_for_cer):
    #         cur_sentences_options = []
    #         for idx, norm_options in enumerate(cur_sent):
    #             best_option, cer, best_idx = self.select_best_match(
    #                 normalized_texts=norm_options,
    #                 input_text=" ".join(text_list[sent_idx]),
    #                 pred_text=pred_text,
    #                 verbose=verbose,
    #                 cer_threshold=-1,
    #             )
    #             cur_sentences_options.append(audio_based_options[sent_idx][idx][best_idx])
    #         selected_options.append(cur_sentences_options)
    #
    #     normalized_text = ""
    #     for sent_idx in range(len(text_list)):
    #         for i, semiotic_idx in enumerate(masked_idx_list[sent_idx]):
    #             text_list[sent_idx][semiotic_idx] = selected_options[sent_idx][i]
    #
    #         normalized_text += " " + " ".join(text_list[sent_idx])
    #     return normalized_text.replace("  ", " ")

    def normalize(
        self,
        text: str,
        n_tagged: int,
        punct_post_process: bool = True,
        verbose: bool = False,
        pred_text: str = None,
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

        Returns:
            normalized text options (usually there are multiple ways of normalizing a given semiotic class)
        """
        #################################
        # LONG AUDIO WITH DIFF APPROACH
        #################################

        det_norm = super().normalize(
            text=text, verbose=verbose, punct_pre_process=False, punct_post_process=punct_post_process
        )
        semiotic_spans, pred_text_spans, norm_spans, text_with_span_tags_list, masked_idx_list = get_alignment(
            text, det_norm, pred_text, verbose=False
        )

        sem_tag_idx = 0
        for cur_semiotic_span, cur_pred_text in zip(semiotic_spans, pred_text_spans):
            if len(cur_semiotic_span) == 0:
                text_with_span_tags_list[masked_idx_list[sem_tag_idx]] = ""
            else:
                non_deter_options = self.normalize_non_deterministic(
                    text=cur_semiotic_span, n_tagged=n_tagged, punct_post_process=punct_post_process, verbose=verbose,
                )
                try:
                    best_option, cer, best_idx = self.select_best_match(
                        normalized_texts=non_deter_options,
                        input_text=cur_semiotic_span,
                        pred_text=cur_pred_text,
                        verbose=verbose,
                        cer_threshold=-1,
                    )
                except:
                    import pdb

                    pdb.set_trace()
                    print()
                text_with_span_tags_list[masked_idx_list[sem_tag_idx]] = best_option

            sem_tag_idx += 1

        normalized_text = " ".join(text_with_span_tags_list)
        return normalized_text.replace("  ", " ")

    def normalize_non_deterministic(
        self, text: str, n_tagged: int, punct_post_process: bool = True, verbose: bool = False
    ):
        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text
        text = pynini.escape(text)
        text = pre_process(text)
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
                self._verbalize(tagged_text, normalized_texts, verbose=verbose)

        if len(normalized_texts) == 0:
            raise ValueError()

        if punct_post_process:
            # do post-processing based on Moses detokenizer
            if self.processor:
                normalized_texts = [self.processor.detokenize([t]) for t in normalized_texts]
                normalized_texts = [post_process_punct(input=text, normalized_text=t) for t in normalized_texts]

        if self.lm:
            remove_dup = sorted(list(set(zip(normalized_texts, weights))), key=lambda x: x[1])
            normalized_texts, weights = zip(*remove_dup)
            return list(normalized_texts), weights

        normalized_texts = set(normalized_texts)
        return normalized_texts

    def normalize_line(
        self,
        n_tagged,
        line: str,
        verbose: bool = False,
        punct_pre_process=False,
        punct_post_process=True,
        text_field: str = "text",
        output_field: str = "normalized",
        **kwargs,
    ):
        line = json.loads(line)

        # TODO add these fields to the args
        asr_pred_field = kwargs.get("asr_pred_field", "pred_text")

        normalized_text = self.normalize(
            text=line["text"],
            verbose=verbose,
            n_tagged=n_tagged,
            punct_post_process=punct_post_process,
            pred_text=line[asr_pred_field],
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

    def _verbalize(self, tagged_text: str, normalized_texts: List[str], verbose: bool = False):
        """
        Verbalizes tagged text

        Args:
            tagged_text: text with tags
            normalized_texts: list of possible normalization options
            verbose: if true prints intermediate classification results
        """

        def get_verbalized_text(tagged_text):
            return rewrite.rewrites(tagged_text, self.verbalizer.fst)

        self.parser(tagged_text)
        tokens = self.parser.parse()
        tags_reordered = self.generate_permutations(tokens)
        for tagged_text_reordered in tags_reordered:
            try:
                tagged_text_reordered = pynini.escape(tagged_text_reordered)
                normalized_texts.extend(get_verbalized_text(tagged_text_reordered))
                if verbose:
                    print(tagged_text_reordered)

            except pynini.lib.rewrite.Error:
                continue

    def select_best_match(
        self,
        normalized_texts: List[str],
        input_text: str,
        pred_text: str,
        verbose: bool = False,
        remove_punct: bool = False,
        cer_threshold: int = 100,
    ):
        """
        Selects the best normalization option based on the lowest CER

        Args:
            normalized_texts: normalized text options
            input_text: input text
            pred_text: ASR model transcript of the audio file corresponding to the normalized text
            verbose: whether to print intermediate meta information
            remove_punct: whether to remove punctuation before calculating CER
            cer_threshold: if CER for pred_text is above the cer_threshold, no normalization will be performed.
                Set to -1 to disable cer-based filtering

        Returns:
            normalized text with the lowest CER and CER value
        """
        if pred_text == "":
            return input_text, cer_threshold

        normalized_texts_cer = calculate_cer(normalized_texts, pred_text, remove_punct)
        normalized_texts_cer = sorted(normalized_texts_cer, key=lambda x: x[1])
        normalized_text, cer, idx = normalized_texts_cer[-1]

        if cer_threshold > 0 and cer > cer_threshold:
            return input_text, cer

        if verbose:
            print('-' * 30)
            for option in normalized_texts:
                print(option)
            print('-' * 30)
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

        cer = Levenshtein.ratio(text_clean, pred_text)
        # cer = round(word_error_rate([pred_text], [text_clean], use_cer=True) * 100, 2)
        normalized_options.append((text, cer, i))
    return normalized_options


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--text", help="input string or path to a .txt file", default=None, type=str)
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument(
        "--language", help="Select target language", choices=["en", "ru", "de", "es"], default="en", type=str
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
        default=100,
        type=int,
        help="if CER for pred_text is above the cer_threshold, no normalization will be performed",
    )
    parser.add_argument("--batch_size", default=200, type=int, help="Number of examples for each process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # if not ASR_AVAILABLE and args.manifest:
    #     raise ValueError("NeMo ASR collection is not installed.")

    args.whitelist = os.path.abspath(args.whitelist) if args.whitelist else None
    if args.text is not None:
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
            lm=args.lm,
        )
        start = perf_counter()
        if os.path.exists(args.text):
            with open(args.text, 'r') as f:
                args.text = f.read().strip()

        options = normalizer.normalize_non_deterministic(
            text=args.text,
            n_tagged=args.n_tagged,
            punct_post_process=not args.no_punct_post_process,
            verbose=args.verbose,
        )
        for option in options:
            print(option)
    elif args.manifest.endswith('.json'):
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
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
            pred_text=args.manifest_asr_pred_field,
        )
    else:
        raise ValueError(
            "Provide either path to .json manifest with '--manifest' OR "
            + "an input text with '--text' (for debugging without audio)"
        )
    print(f'Execution time: {round((perf_counter() - start)/60, 2)} min.')
