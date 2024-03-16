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

import os
from argparse import ArgumentParser
from time import perf_counter
from typing import List, Dict, Tuple
import re

from nemo_text_processing.text_normalization.data_loader_utils import load_file, write_file
from nemo_text_processing.text_normalization.en.graph_utils import INPUT_CASED, INPUT_LOWER_CASED
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser


class InverseNormalizer(Normalizer):
    """
    Inverse normalizer that converts text from spoken to written form. Useful for ASR postprocessing.
    Input is expected to have no punctuation outside of approstrophe (') and dash (-) and be lower cased.

    Args:
        input_case: Input text capitalization, set to 'cased' if text contains capital letters.
            This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.
        lang: language specifying the ITN
        whitelist: path to a file with whitelist replacements. (each line of the file: written_form\tspoken_form\n),
            e.g. nemo_text_processing/inverse_text_normalization/en/data/whitelist.tsv
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        max_number_of_permutations_per_split: a maximum number
            of permutations which can be generated from input sequence of tokens.
    """

    def __init__(
        self,
        input_case: str = INPUT_LOWER_CASED,
        lang: str = "en",
        whitelist: str = None,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        max_number_of_permutations_per_split: int = 729,
    ):

        assert input_case in ["lower_cased", "cased"]

        if lang == 'en':  # English
            from nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'es':  # Spanish (Espanol)
            from nemo_text_processing.inverse_text_normalization.es.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.es.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'pt':  # Portuguese (Português)
            from nemo_text_processing.inverse_text_normalization.pt.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.pt.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'ru':  # Russian (Russkiy Yazyk)
            from nemo_text_processing.inverse_text_normalization.ru.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.ru.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'de':  # German (Deutsch)
            from nemo_text_processing.inverse_text_normalization.de.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.de.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'fr':  # French (Français)
            from nemo_text_processing.inverse_text_normalization.fr.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.fr.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'sv':  # Swedish (Svenska)
            from nemo_text_processing.inverse_text_normalization.sv.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.sv.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'vi':  # Vietnamese (Tiếng Việt)
            from nemo_text_processing.inverse_text_normalization.vi.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.vi.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'ar':  # Arabic
            from nemo_text_processing.inverse_text_normalization.ar.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.ar.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'es_en':  # Arabic
            from nemo_text_processing.inverse_text_normalization.es_en.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.es_en.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'zh':  # Mandarin
            from nemo_text_processing.inverse_text_normalization.zh.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.zh.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'mr':  # Marathi
            from nemo_text_processing.inverse_text_normalization.mr.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.mr.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'hy':
            from nemo_text_processing.inverse_text_normalization.hy.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.hy.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        self.numbers_tagger = ClassifyFst(
            cache_dir=cache_dir, 
            whitelist=whitelist, 
            overwrite_cache=overwrite_cache, 
            input_case=input_case,
            classify_number=True,
            classify_date=True,
            classify_time=True,
            classify_money=True,
            classify_telephone=True,
            classify_measure=False,
            classify_whitelist=False,
            classify_electronic=False,
        )
        
        self.other_tagger = ClassifyFst(
            cache_dir=cache_dir, 
            whitelist=whitelist, 
            overwrite_cache=overwrite_cache, 
            input_case=input_case,
            classify_number=False,
            classify_date=False,
            classify_time=False,
            classify_money=False,
            classify_telephone=False,
            classify_measure=True,
            classify_whitelist=True,
            classify_electronic=True,
        )      
        
        self.default_tagger = ClassifyFst(
            cache_dir=cache_dir, 
            whitelist=whitelist, 
            overwrite_cache=overwrite_cache, 
            input_case=input_case,
        )
        self.verbalizer = VerbalizeFinalFst()
        self.parser = TokenParser()
        self.lang = lang
        self.input_case = input_case
        self.word_to_symbol, self.symbol_to_word = self.symbol_mapping()
        self.probable_email_pattern: re.Pattern = self.get_probable_email_regex_pattern()
        self.email_prefix_pattern: re.Pattern = self.get_email_prompt_regex_pattern()
        self.tuple_to_value = self.get_tuple_to_value()
        self.tuple_before_number_regex_pattern: re.Pattern = self.get_tuple_before_number_regex_pattern()
        self.max_number_of_permutations_per_split = max_number_of_permutations_per_split

    def inverse_normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo inverse text normalizer

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list of input strings
        """
        return self.normalize_list(texts=texts, verbose=verbose)

    def inverse_normalize(self, text: str, verbose: bool) -> str:
        """
        Main function. Inverse normalizes tokens from spoken to written form
            e.g. twelve kilograms -> 12 kg

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: written form
        """
        inverse_normalized: str
        if self.input_case == "lower_cased":
            probable_email = bool(self.probable_email_pattern.search(text, re.IGNORECASE))
        else:
            probable_email = bool(self.probable_email_pattern.search(text))
        if not probable_email:
            self.tagger = self.default_tagger
            tuples_normalized: str = self.normalize_tuples_before_numbers(
                text=text,
                pattern=self.tuple_before_number_regex_pattern,
                tuple_dict=self.tuple_to_value,
            )
            inverse_normalized = self.normalize(text=tuples_normalized, verbose=verbose)
        else:
            self.tagger = self.numbers_tagger
            numbers_inverse_normalized = self.normalize(text=text, verbose=verbose)
            self.tagger = self.other_tagger
            inverse_normalized = self.normalize(text=numbers_inverse_normalized, verbose=verbose)
            inverse_normalized = self.process_email(
                unprocessed_text=numbers_inverse_normalized,
                inverse_normalized_text=inverse_normalized,
            )
        
        return inverse_normalized
    
    def get_probable_email_regex_pattern(self) -> re.Pattern:
        probable_email_pattern = r"".join([
            "(?:\w+ +)+(?:",
            "|".join(self.symbol_to_word['@']),
            ") +(?:\w+ +)+(?:",
            "|".join(self.symbol_to_word['.']),
            ") +\w",
        ])
        if self.input_case == "lower_cased":
            pattern_regex = re.compile(probable_email_pattern, re.IGNORECASE)
        else:
            pattern_regex = re.compile(probable_email_pattern)
            
        return pattern_regex
        
    def get_email_prompt_regex_pattern(self) -> re.Pattern:
        pattern_list = load_file(self.get_abs_path(f"{self.lang}/data/electronic/email_prompt_pattern.tsv"))
        pattern_list = [pattern.rstrip('\n') for pattern in pattern_list]
        if self.input_case == "lower_cased":
            pattern_regex = re.compile(r"\b"+"|".join(pattern_list) + r"\b", re.IGNORECASE)
        else:
            pattern_regex = re.compile(r"\b"+"|".join(pattern_list) + r"\b")
        return pattern_regex
    
    def symbol_mapping(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        word_to_symbol_mapping: Dict[str, str] = {}
        symbol_to_word_mapping: Dict[str, List[str]] = {}
        symbols_list = load_file(self.get_abs_path(f"{self.lang}/data/electronic/symbols.tsv"))
        for symbol in symbols_list:
            symbol_char, symbol_name = symbol.rstrip('\n').split('\t')
            word_to_symbol_mapping[symbol_name] = symbol_char
            if symbol_to_word_mapping.get(symbol_char) is None:
                symbol_to_word_mapping[symbol_char] = [symbol_name]
            else:
                symbol_to_word_mapping[symbol_char].append(symbol_name)
        
        return (word_to_symbol_mapping, symbol_to_word_mapping)
    
    def process_email(self, unprocessed_text: str, inverse_normalized_text: str) -> str:
    
        for i, token in enumerate(reversed(self.tokens), 1):
            electronic_token = token['tokens'].get('electronic')
            if electronic_token is not None:
                word_splitted_text = inverse_normalized_text.split(' ')
                electronic_text: str = word_splitted_text[-i]
                # print(electronic_text)
                username_offset = 0
                if '@' not in electronic_text:
                    for symbol_word_length in range(1,4):
                        if " ".join(word_splitted_text[-i-symbol_word_length:-i]) in self.symbol_to_word['@']:
                            electronic_text = "@" + electronic_text
                            username_offset = symbol_word_length
                            break
                    if username_offset == 0:
                        break
                email_username_words = word_splitted_text[:-i-username_offset]
                
                for j, word in enumerate(email_username_words):
                    if word != unprocessed_text.split(' ')[j]:
                        for k, unprocessed_word in enumerate(unprocessed_text.split(' ')[j:]):
                            if word.endswith(unprocessed_word):
                                email_username_words = unprocessed_text.split(' ')[:j+k+1]
                                break
                
                prefix_search_string = " ".join(email_username_words)
                matched_prefix_list = list(self.email_prefix_pattern.finditer(prefix_search_string))
                prefix_string = ""
  
                if matched_prefix_list:
                    email_username_words = prefix_search_string[matched_prefix_list[-1].regs[0][1]:].split(' ')
                    prefix_string = prefix_search_string[:matched_prefix_list[-1].regs[0][1]]

                for word in reversed(email_username_words):
                    if word is not None:
                        word = self.word_to_symbol.get(word) if self.word_to_symbol.get(word) is not None else word
                        electronic_text = word + electronic_text
                
                return " ".join([prefix_string, electronic_text] + word_splitted_text[len(word_splitted_text)-i+1:])
        
        return inverse_normalized_text
    
    def normalize_tuples_before_numbers(self, text: str, pattern:re.Pattern, tuple_dict: Dict[str, int]) -> str:
        normalized_text: str = pattern.sub(lambda m: f"{m.group(2)} " * tuple_dict[m.group(1)], text)
        return normalized_text
        
        
    def get_tuple_to_value(self) -> Dict[str, int]:
        tuple_to_value: Dict[str, int] = {}
        tuple_name_list = load_file(self.get_abs_path(f"{self.lang}/data/numbers/tuples.tsv"))
        for tuple_name in tuple_name_list:
            tuple_term, tuple_value = tuple_name.rstrip('\n').split('\t')
            tuple_to_value[tuple_term] = int(tuple_value)
        
        return tuple_to_value
    
    def get_tuple_before_number_regex_pattern(self) -> re.Pattern:
        numbers_to_value: Dict[str, int] = {}
        digits_list = load_file(self.get_abs_path(f"{self.lang}/data/numbers/digit.tsv"))
        for digit in digits_list:
            digit_term, digit_value = digit.rstrip('\n').split('\t')
            numbers_to_value[digit_term] = digit_value
            
        teen_numbers_list = load_file(self.get_abs_path(f"{self.lang}/data/numbers/teen.tsv"))
        for teen_number in teen_numbers_list:
            teen_number_term, teen_number_value = teen_number.rstrip('\n').split('\t')
            numbers_to_value[teen_number_term] = teen_number_value
        
        ties_numbers_list = load_file(self.get_abs_path(f"{self.lang}/data/numbers/ties.tsv"))
        for ties_number in ties_numbers_list:
            ties_number_term, ties_number_value = ties_number.rstrip('\n').split('\t')
            numbers_to_value[ties_number_term] = ties_number_value
            
        zero_number_list = load_file(self.get_abs_path(f"{self.lang}/data/numbers/zero.tsv"))
        for zero_number in zero_number_list:
            zero_number_term, zero_number_value = zero_number.rstrip('\n').split('\t')
            numbers_to_value[zero_number_term] = zero_number_value
        numbers_to_value['o'] = 0
        numbers_to_value['oh'] = 0
        
        tuple_before_number_pattern: str = r"".join([
            r"\b(",
            r"|".join(list(self.get_tuple_to_value().keys())),
            r")\s+(",
            r"|".join(numbers_to_value.keys()),
            r")(?:s?)\b",
        ])
        
        if self.input_case == "lower_cased":
            tuple_before_number_pattern_regex = re.compile(tuple_before_number_pattern, re.IGNORECASE)
        else:
            tuple_before_number_pattern_regex = re.compile(tuple_before_number_pattern)
        return tuple_before_number_pattern_regex

    def get_abs_path(self, rel_path):
        """
        Get absolute path

        Args:
            rel_path: relative path to this file
            
        Returns absolute path
        """
        return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path

def parse_args():
    parser = ArgumentParser()
    input = parser.add_mutually_exclusive_group()
    input.add_argument("--text", dest="input_string", help="input string", type=str)
    input.add_argument("--input_file", dest="input_file", help="input file path", type=str)
    parser.add_argument('--output_file', dest="output_file", help="output file path", type=str)
    parser.add_argument(
        "--language",
        help="language",
        choices=['en', 'de', 'es', 'pt', 'ru', 'fr', 'sv', 'vi', 'ar', 'es_en', 'zh', 'hy', 'mr'],
        default="en",
        type=str,
    )
    parser.add_argument(
        "--input_case",
        help="Input text capitalization, set to 'cased' if text contains capital letters."
        "This flag affects normalization rules applied to the text. Note, `lower_cased` won't lower case input.",
        choices=[INPUT_CASED, INPUT_LOWER_CASED],
        default=INPUT_LOWER_CASED,
        type=str,
    )
    parser.add_argument(
        "--whitelist",
        help="Path to a file with with whitelist replacements," "e.g., inverse_normalization/en/data/whitelist.tsv",
        default=None,
        type=str,
    )
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument("--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true")
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    whitelist = os.path.abspath(args.whitelist) if args.whitelist else None
    start_time = perf_counter()
    inverse_normalizer = InverseNormalizer(
        input_case=args.input_case,
        lang=args.language,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        whitelist=whitelist,
    )
    print(f'Time to generate graph: {round(perf_counter() - start_time, 2)} sec')

    if args.input_string:
        print(inverse_normalizer.inverse_normalize(args.input_string, verbose=args.verbose))
    elif args.input_file:
        print("Loading data: " + args.input_file)
        data = load_file(args.input_file)

        print("- Data: " + str(len(data)) + " sentences")
        prediction = inverse_normalizer.inverse_normalize_list(data, verbose=args.verbose)
        if args.output_file:
            write_file(args.output_file, prediction)
            print(f"- Denormalized. Writing out to {args.output_file}")
        else:
            print(prediction)
