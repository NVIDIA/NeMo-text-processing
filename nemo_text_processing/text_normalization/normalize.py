# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import json
import os
import re
import shutil
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from math import factorial
from time import perf_counter
from typing import Dict, List, Optional, Union

import pynini
import regex
import tqdm
from joblib import Parallel, delayed
from nemo_text_processing.logging import logger
from nemo_text_processing.text_normalization.data_loader_utils import (
    load_file,
    post_process_punct,
    pre_process,
    write_file,
)
from nemo_text_processing.text_normalization.preprocessing_utils import additional_split
from nemo_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from pynini.lib.rewrite import top_rewrite
from sacremoses import MosesDetokenizer
from tqdm import tqdm

# this is to handle long input
sys.setrecursionlimit(3000)

SPACE_DUP = re.compile(' {2,}')


"""
To normalize a single entry:
    python normalize.py --text=<INPUT_TEXT>
        
To normalize text in .json manifest:

    python normalize.py \
        --input_file=<PATH TO INPUT .JSON MANIFEST> \
        --output_file=<PATH TO OUTPUT .JSON MANIFEST> \
        --n_jobs=-1 \
        --batch_size=300 \
        --manifest_text_field="text" \
        --whitelist=<PATH TO YOUR WHITELIST>
    
    For a complete list of optional arguments, run:
    >>> python normalize.py --help


To integrate Normalizer in your script:
    >>> from nemo_text_processing.text_normalization.normalize import Normalizer
    # see the script for args details
    >>> normalizer_en = Normalizer(input_case='cased', lang='en', cache_dir=CACHE_DIR, overwrite_cache=False, post_process=True)
    >>> normalizer_en.normalize("<INPUT_TEXT>")
    # normalize list of entries
    >>> normalizer_en.normalize_list(["<INPUT_TEXT1>", "<INPUT_TEXT2>"])
    # normalize .json manifest entries
    >>> normalizer_en.normalize_manifest(manifest=<PATH TO INPUT .JSON MANIFEST>, n_jobs=-1, batch_size=300, 
                                        output_filename=<PATH TO OUTPUT .JSON MANIFEST>, text_field="text",
                                        punct_pre_process=False, punct_post_process=False)

"""


class Normalizer:
    """
    Normalizer class that converts text from written to spoken form.
    Useful for TTS preprocessing.

    Args:
        input_case: Input text capitalization, set to 'cased' if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
        lang: language specifying the TN rules, by default: English
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        post_process: WFST-based post processing, e.g. to remove extra spaces added during TN.
            Note: punct_post_process flag in normalize() supports all languages.
        max_number_of_permutations_per_split: a maximum number
            of permutations which can be generated from input sequence of tokens.
        verbose: whether to print intermediate meta information
    """

    def __init__(
        self,
        input_case: str,
        lang: str = 'en',
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        lm: bool = False,
        post_process: bool = True,
        max_number_of_permutations_per_split: int = 729,
    ):
        assert input_case in ["lower_cased", "cased"]

        self.post_processor = None

        if lang == "en":
            from nemo_text_processing.text_normalization.en.verbalizers.verbalize_final import VerbalizeFinalFst
            from nemo_text_processing.text_normalization.en.verbalizers.post_processing import PostProcessingFst

            if post_process:
                self.post_processor = PostProcessingFst(cache_dir=cache_dir, overwrite_cache=overwrite_cache)

            if deterministic:
                from nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
            else:
                if lm:
                    from nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify_lm import ClassifyFst
                else:
                    from nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify_with_audio import (
                        ClassifyFst,
                    )
        elif lang == 'ru':
            # Ru TN only support non-deterministic cases and produces multiple normalization options
            # use normalize_with_audio.py
            from nemo_text_processing.text_normalization.ru.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.ru.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'de':
            from nemo_text_processing.text_normalization.de.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.de.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'es':
            from nemo_text_processing.text_normalization.es.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.es.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'fr':
            from nemo_text_processing.text_normalization.fr.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.fr.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'sv':
            from nemo_text_processing.text_normalization.sv.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.sv.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'hu':
            from nemo_text_processing.text_normalization.hu.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.hu.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'zh':
            from nemo_text_processing.text_normalization.zh.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.zh.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'ar':
            from nemo_text_processing.text_normalization.ar.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.ar.verbalizers.verbalize_final import VerbalizeFinalFst
        elif lang == 'it':
            from nemo_text_processing.text_normalization.it.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.text_normalization.it.verbalizers.verbalize_final import VerbalizeFinalFst
        else:
            raise NotImplementedError(f"Language {lang} has not been supported yet.")

        self.input_case = input_case
        self.tagger = ClassifyFst(
            input_case=self.input_case,
            deterministic=deterministic,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
        )

        self.verbalizer = VerbalizeFinalFst(
            deterministic=deterministic, cache_dir=cache_dir, overwrite_cache=overwrite_cache
        )
        self.max_number_of_permutations_per_split = max_number_of_permutations_per_split
        self.parser = TokenParser()
        self.lang = lang
        self.moses_detokenizer = MosesDetokenizer(lang=lang)

    def normalize_list(
        self,
        texts: List[str],
        verbose: bool = False,
        punct_pre_process: bool = False,
        punct_post_process: bool = False,
        batch_size: int = 1,
        n_jobs: int = 1,
        **kwargs,
    ):
        """
        NeMo text normalizer

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information
            punct_pre_process: whether to do punctuation pre-processing
            punct_post_process: whether to do punctuation post-processing
            n_jobs: the maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given,
                no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
            batch_size: Number of examples for each process

        Returns converted list input strings
        """

        def _process_batch(batch, verbose, punct_pre_process, punct_post_process, **kwargs):
            """
            Normalizes batch of text sequences
            Args:
                batch: list of texts
                verbose: whether to print intermediate meta information
                punct_pre_process: whether to do punctuation pre-processing
                punct_post_process: whether to do punctuation post-processing
            """
            normalized_lines = [
                self.normalize(
                    text,
                    verbose=verbose,
                    punct_pre_process=punct_pre_process,
                    punct_post_process=punct_post_process,
                    **kwargs,
                )
                for text in tqdm(batch)
            ]
            return normalized_lines

        # to save intermediate results to a file
        batch = min(len(texts), batch_size)

        try:
            normalized_texts = Parallel(n_jobs=n_jobs)(
                delayed(_process_batch)(texts[i : i + batch], verbose, punct_pre_process, punct_post_process, **kwargs)
                for i in range(0, len(texts), batch)
            )
        except BaseException as e:
            raise e

        normalized_texts = list(itertools.chain(*normalized_texts))
        return normalized_texts

    def _estimate_number_of_permutations_in_nested_dict(
        self, token_group: Dict[str, Union[OrderedDict, str, bool]]
    ) -> int:
        num_perms = 1
        for k, inner in token_group.items():
            if isinstance(inner, dict):
                num_perms *= self._estimate_number_of_permutations_in_nested_dict(inner)
        num_perms *= factorial(len(token_group))
        return num_perms

    def _split_tokens_to_reduce_number_of_permutations(self, tokens: List[dict]) -> List[List[dict]]:
        """
        Splits a sequence of tokens in a smaller sequences of tokens in a way that maximum number of composite
        tokens permutations does not exceed ``max_number_of_permutations_per_split``.

        For example,

        .. code-block:: python

            # setup normalizer with self.max_number_of_permutations_per_split=6
             tokens = [
                {"tokens": {"date": {"year": "twenty eighteen", "month": "december", "day": "thirty one"}}},
                {"tokens": {"date": {"year": "twenty eighteen", "month": "january", "day": "eight"}}},
            ]
            split = normalizer._split_tokens_to_reduce_number_of_permutations(tokens)
            assert split == [
                [{"tokens": {"date": {"year": "twenty eighteen", "month": "december", "day": "thirty one"}}}],
                [{"tokens": {"date": {"year": "twenty eighteen", "month": "january", "day": "eight"}}}],
            ]

        Date tokens contain 3 items each which gives 6 permutations for every date. Since there are 2 dates, total
        number of permutations would be ``6 * 6 == 36``. Parameter ``self.max_number_of_permutations_per_split`` equals 6,
        so input sequence of tokens is split into 2 smaller sequences.

        Args:
            tokens: a list of dictionaries, possibly nested.

        Returns:
            a list of smaller sequences of tokens resulting from ``tokens`` split.
        """
        splits = []
        prev_end_of_split = 0
        current_number_of_permutations = 1
        for i, token_group in enumerate(tokens):
            n = self._estimate_number_of_permutations_in_nested_dict(token_group)
            if n * current_number_of_permutations > self.max_number_of_permutations_per_split:
                splits.append(tokens[prev_end_of_split:i])
                prev_end_of_split = i
                current_number_of_permutations = 1
            if n > self.max_number_of_permutations_per_split:
                raise ValueError(
                    f"Could not split token list with respect to condition that every split can generate number of "
                    f"permutations less or equal to "
                    f"`self.max_number_of_permutations_per_split={self.max_number_of_permutations_per_split}`. "
                    f"There is an unsplittable token group that generates more than "
                    f"{self.max_number_of_permutations_per_split} permutations. Try to increase "
                    f"`--max_number_of_permutations_per_split` parameter."
                )
            current_number_of_permutations *= n
        splits.append(tokens[prev_end_of_split:])
        assert sum([len(s) for s in splits]) == len(tokens)
        return splits

    def normalize(
        self, text: str, verbose: bool = False, punct_pre_process: bool = False, punct_post_process: bool = False
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
            punct_post_process: whether to normalize punctuation
            verbose: whether to print intermediate meta information

        Returns: spoken form
        """
        if len(text.split()) > 500:
            logger.warning(
                "Your input is too long and could take a long time to normalize. "
                "Use split_text_into_sentences() to make the input shorter and then call normalize_list()."
            )
        original_text = text
        if punct_pre_process:
            text = pre_process(text)
        text = text.strip()
        if not text:
            if verbose:
                logger.info(text)
            return text
        text = pynini.escape(text)
        tagged_lattice = self.find_tags(text)
        tagged_text = Normalizer.select_tag(tagged_lattice)
        if verbose:
            logger.info(tagged_text)
        self.parser(tagged_text)
        tokens = self.parser.parse()
        split_tokens = self._split_tokens_to_reduce_number_of_permutations(tokens)
        output = ""
        for s in split_tokens:
            try:
                tags_reordered = self.generate_permutations(s)
                verbalizer_lattice = None
                for tagged_text in tags_reordered:
                    tagged_text = pynini.escape(tagged_text)

                    verbalizer_lattice = self.find_verbalizer(tagged_text)
                    if verbalizer_lattice.num_states() != 0:
                        break
                if verbalizer_lattice is None:
                    logger.warning(f"No permutations were generated from tokens {s}")
                    return text
                output += ' ' + Normalizer.select_verbalizer(verbalizer_lattice)
            except Exception as e:
                logger.warning("Failed text: " + text + str(e))
                return text
        output = SPACE_DUP.sub(' ', output[1:])

        if self.lang == "en" and hasattr(self, 'post_processor'):
            output = self.post_process(output)

        if punct_post_process:
            # do post-processing based on Moses detokenizer
            output = self.moses_detokenizer.detokenize([output], unescape=False)
            output = post_process_punct(input=original_text, normalized_text=output)
        return output

    def normalize_line(
        self,
        line: str,
        verbose: bool = False,
        punct_pre_process=False,
        punct_post_process=True,
        text_field: str = "text",
        output_field: str = "normalized",
        **kwargs,
    ):
        """
        Normalizes "text_field" in line from a .json manifest

        Args:
            line: line of a .json manifest
            verbose: set to True to see intermediate output of normalization
            punct_pre_process: set to True to do punctuation pre-processing
            punct_post_process: set to True to do punctuation post-processing
            text_field: name of the field in the manifest to normalize
            output_field: name of the field in the manifest to save normalized text
        """
        line = json.loads(line)

        normalized_text = self.normalize(
            text=line[text_field],
            verbose=verbose,
            punct_pre_process=punct_pre_process,
            punct_post_process=punct_post_process,
            **kwargs,
        )
        line[output_field] = normalized_text
        return line

    def normalize_manifest(
        self,
        manifest: str,
        n_jobs: int,
        punct_pre_process: bool,
        punct_post_process: bool,
        batch_size: int,
        output_filename: Optional[str] = None,
        text_field: str = "text",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Normalizes "text_field" from .json manifest.

        Args:
            manifest: path to .json manifest file
            n_jobs: the maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given,
                no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
            punct_pre_process: set to True to do punctuation pre-processing
            punct_post_process: set to True to do punctuation post-processing
            batch_size: number of samples to process per iteration (int)
            output_filename: path to .json file to save normalized text
            text_field: name of the field in the manifest to normalize
            **kwargs are need for audio-based normalization that requires extra args
        """

        def _process_batch(
            batch_idx: int,
            batch: List[str],
            dir_name: str,
            verbose=verbose,
            punct_pre_process=False,
            punct_post_process=True,
            text_field: str = "text",
            output_field: str = "normalized",
            **kwargs,
        ):
            """
            Normalizes batch of text sequences
            Args:
                batch: list of texts
                batch_idx: batch index
                dir_name: path to output directory to save results
            """
            normalized_lines = [
                self.normalize_line(
                    line=line,
                    verbose=verbose,
                    punct_post_process=punct_post_process,
                    punct_pre_process=punct_pre_process,
                    text_field=text_field,
                    output_field=output_field,
                    **kwargs,
                )
                for line in tqdm(batch)
            ]

            with open(f"{dir_name}/{batch_idx:06}.json", "w") as f_out:
                for line in normalized_lines:
                    if isinstance(line[output_field], set):
                        if len(line[output_field]) > 1:
                            logger.warning("Len of " + str(line[output_field]) + " > 1 ")
                        line[output_field] = line[output_field].pop()

                    f_out.write(json.dumps(line, ensure_ascii=False) + '\n')

            logger.info(f"Batch -- {batch_idx} -- is complete")

        if output_filename is None:
            output_filename = manifest.replace('.json', '_normalized.json')

        with open(manifest, 'r') as f:
            lines = f.readlines()

        logger.warning(f'Normalizing {len(lines)} line(s) of {manifest}...')

        # to save intermediate results to a file
        batch = min(len(lines), batch_size)

        tmp_dir = "/tmp/parts"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        Parallel(n_jobs=n_jobs)(
            delayed(_process_batch)(
                idx,
                lines[i : i + batch],
                tmp_dir,
                text_field=text_field,
                verbose=verbose,
                punct_pre_process=punct_pre_process,
                punct_post_process=punct_post_process,
                **kwargs,
            )
            for idx, i in enumerate(range(0, len(lines), batch))
        )

        # aggregate all intermediate files
        with open(output_filename, "w") as f_out:
            for batch_f in sorted(glob(f"{tmp_dir}/*.json")):
                with open(batch_f, "r") as f_in:
                    lines = f_in.read()
                    f_out.write(lines)

        logger.warning(f'Normalized version saved at {output_filename}')

    def split_text_into_sentences(self, text: str, additional_split_symbols: str = "") -> List[str]:
        """
        Split text into sentences.

        Args:
            text: text
            additional_split_symbols: Symbols to split sentences if eos sentence split resulted in a long sequence.
                Use '|' as a separator between symbols, for example: ';|:'. Use '\s' to split by space.

        Returns list of sentences
        """
        lower_case_unicode = ""
        upper_case_unicode = ""

        if self.lang == "ru":
            lower_case_unicode = '\u0430-\u04FF'
            upper_case_unicode = '\u0410-\u042F'

        # end of quoted speech - to be able to split sentences by full stop
        text = re.sub(r"([\.\?\!])([\"\'])", r"\g<2>\g<1> ", text)

        # remove extra space
        text = re.sub(r" +", " ", text)

        # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
        matches = re.findall(rf"[a-z{lower_case_unicode}]\.\s[a-z{lower_case_unicode}]\.", text)
        for match in matches:
            text = text.replace(match, match.replace(". ", "."))

        # Read and split transcript by utterance (roughly, sentences)
        split_pattern = rf"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]+\.)(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s(?![0-9]+[a-z]*\.)"
        sentences = regex.split(split_pattern, text)
        sentences = additional_split(sentences, additional_split_symbols)
        return sentences

    def _permute(self, d: OrderedDict) -> List[str]:
        """
        Creates reorderings of dictionary elements and serializes as strings

        Args:
            d: (nested) dictionary of key value pairs

        Return permutations of different string serializations of key value pairs
        """
        l = []
        if PRESERVE_ORDER_KEY in d.keys():
            d_permutations = [d.items()]
        else:
            d_permutations = itertools.permutations(d.items())
        for perm in d_permutations:
            subl = [""]
            for k, v in perm:
                if isinstance(v, str):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: \"{v}\" "])]
                elif isinstance(v, OrderedDict):
                    rec = self._permute(v)
                    subl = ["".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])]
                elif isinstance(v, bool):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: true "])]
                else:
                    raise ValueError("Key: " + str(k) + " Value: " + str(v))
            l.extend(subl)
        return l

    def generate_permutations(self, tokens: List[dict]):
        """
        Generates permutations of string serializations of list of dictionaries

        Args:
            tokens: list of dictionaries

        Returns string serialization of list of dictionaries
        """

        def _helper(prefix: str, token_list: List[dict], idx: int):
            """
            Generates permutations of string serializations of given dictionary

            Args:
                token_list: list of dictionaries
                prefix: prefix string
                idx:    index of next dictionary

            Returns string serialization of dictionary
            """
            if idx == len(token_list):
                yield prefix
                return
            token_options = self._permute(token_list[idx])
            for token_option in token_options:
                yield from _helper(prefix + token_option, token_list, idx + 1)

        return _helper("", tokens, 0)

    def find_tags(self, text: str) -> 'pynini.FstLike':
        """
        Given text use tagger Fst to tag text

        Args:
            text: sentence

        Returns: tagged lattice
        """
        lattice = text @ self.tagger.fst
        return lattice

    @staticmethod
    def select_tag(lattice: 'pynini.FstLike') -> str:
        """
        Given tagged lattice return shortest path

        Args:
            lattice: pynini.FstLike tag lattice

        Returns: shortest path
        """
        tagged_text = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return tagged_text

    def find_verbalizer(self, tagged_text: str) -> 'pynini.FstLike':
        """
        Given tagged text creates verbalization lattice
        This is context-independent.

        Args:
            tagged_text: input text

        Returns: verbalized lattice
        """
        lattice = tagged_text @ self.verbalizer.fst
        return lattice

    @staticmethod
    def select_verbalizer(lattice: 'pynini.FstLike') -> str:
        """
        Given verbalized lattice return shortest path

        Args:
            lattice: verbalization lattice
            text: full text line to raise in case of an exception

        Returns: shortest path
        """
        output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        # lattice = output @ self.verbalizer.punct_graph
        # output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return output

    def post_process(self, normalized_text: 'pynini.FstLike') -> str:
        """
        Runs post-processing graph on normalized text

        Args:
            normalized_text: normalized text

        Returns: shortest path
        """
        normalized_text = normalized_text.strip()
        if not normalized_text:
            return normalized_text
        normalized_text = pynini.escape(normalized_text)

        if self.post_processor is not None:
            normalized_text = top_rewrite(normalized_text, self.post_processor.fst)
        return normalized_text


def parse_args():
    parser = ArgumentParser()
    input = parser.add_mutually_exclusive_group()
    input.add_argument("--text", dest="input_string", help="input string", type=str)
    input.add_argument(
        "--input_file",
        dest="input_file",
        help="input file path. "
        "The input file can be either a .txt file containing one example for normalization per line or "
        "a .json manifest file. Field to normalize in .json manifest is specified with `--manifest_text_field` arg.",
        type=str,
    )
    parser.add_argument(
        '--manifest_text_field',
        help="The field in a .json manifest to normalize (applicable only when input_file is a .json manifest)",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--output_field",
        help="Name of the field in a .json manifest in which to save normalized text (applicable only when input_file is a .json manifest)",
        type=str,
        default="normalized",
    )
    parser.add_argument('--output_file', dest="output_file", help="Output file path", type=str)
    parser.add_argument(
        "--language",
        help="language",
        choices=["en", "de", "es", "fr", "hu", "sv", "zh", "ar", "it"],
        default="en",
        type=str,
    )
    parser.add_argument(
        "--input_case",
        help="Input text capitalization, set to 'cased' if text contains capital letters."
        "This argument affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.",
        choices=["lower_cased", "cased"],
        default="cased",
        type=str,
    )
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument(
        "--punct_post_process",
        help="Add this flag to enable punctuation post processing to match input.",
        action="store_true",
    )
    parser.add_argument(
        "--punct_pre_process",
        help="Add this flag to add spaces around square brackets, otherwise text between square brackets won't be normalized",
        action="store_true",
    )
    parser.add_argument("--overwrite_cache", help="Add this flag to re-create .far grammar files", action="store_true")
    parser.add_argument(
        "--whitelist",
        help="Path to a file with with whitelist replacement,"
        "e.g., for English, whitelist files are stored under text_normalization/en/data/whitelist",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
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
    whitelist = os.path.abspath(args.whitelist) if args.whitelist else None

    if not args.input_string and not args.input_file:
        raise ValueError("Either `--text` or `--input_file` required")

    normalizer = Normalizer(
        input_case=args.input_case,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        whitelist=whitelist,
        lang=args.language,
        max_number_of_permutations_per_split=args.max_number_of_permutations_per_split,
    )
    start_time = perf_counter()
    if args.input_string:
        logger.info(
            normalizer.normalize(
                args.input_string,
                verbose=args.verbose,
                punct_pre_process=args.punct_pre_process,
                punct_post_process=args.punct_post_process,
            )
        )
    elif args.input_file:
        if args.input_file.endswith(".json"):
            normalizer.normalize_manifest(
                args.input_file,
                n_jobs=args.n_jobs,
                punct_pre_process=args.punct_pre_process,
                punct_post_process=args.punct_post_process,
                batch_size=args.batch_size,
                text_field=args.manifest_text_field,
                output_field=args.output_field,
                output_filename=args.output_file,
                verbose=args.verbose,
            )

        else:
            logger.warning("Loading data: " + args.input_file)
            data = load_file(args.input_file)

            logger.warning("- Data: " + str(len(data)) + " sentences")
            normalizer_prediction = normalizer.normalize_list(
                data,
                verbose=args.verbose,
                punct_pre_process=args.punct_pre_process,
                punct_post_process=args.punct_post_process,
            )
            if args.output_file:
                write_file(args.output_file, normalizer_prediction)
                logger.warning(f"- Normalized. Writing out to {args.output_file}")
            else:
                logger.warning(normalizer_prediction)

    logger.warning(f"Execution time: {perf_counter() - start_time:.02f} sec")
