# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import (
    CURRENCY_SYMBOLS,
    MINUS_WORD,
    NEMO_ALL_DIGIT,
    NEMO_TA_LETTER,
    RANGE_WORD,
    generator_main,
)
from nemo_text_processing.text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ta.taggers.date import DateFst
from nemo_text_processing.text_normalization.ta.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ta.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.ta.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.ta.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.ta.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.ta.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.ta.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.ta.taggers.range import RangeFst
from nemo_text_processing.text_normalization.ta.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.ta.taggers.serial import SerialFst
from nemo_text_processing.text_normalization.ta.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.ta.taggers.time import TimeFst
from nemo_text_processing.text_normalization.ta.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.ta.taggers.word import WordFst
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

# Symbols the whitelist speaks; each is split into its own token wherever it stands.
SPOKEN_SYMBOLS = "#*&^%|~"
# Spoken < and > between two digits, and + between two digits.
LESS_THAN = "விடக் குறைவு"
GREATER_THAN = "விட அதிகம்"
INFIX_PLUS = "கூட்டல்"

# Zero-width and directional format characters with no linguistic role (ZWJ/ZWNJ are kept).
_FORMAT_CHARS = "​﻿⁠­‎‏؜‪‫‬‭‮⁦⁧⁨⁩"
# Dash lookalikes read like an ASCII hyphen; exotic spaces like a space.
_DASHES = "‐‑‒–—―"
_SPACES = "   "


def _pre_process(known_suffixes: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Spacing rewrites composed in front of the sentence grammar: they split symbols and glued
    words off digits and decide what a hyphen or a minus sign means, so that no class grammar
    has to embed those shapes itself.

    Args:
        known_suffixes: every case or ordinal suffix that may stay glued to a digit; any other
            Tamil word glued to a digit is split off
    """
    letter = NEMO_TA_LETTER
    any_digit = NEMO_ALL_DIGIT
    spaces = pynini.closure(" ")
    edge = pynini.union("[BOS]", " ")

    # A zero-width space or word joiner between two digits is a boundary, not glue; the other
    # format characters are dropped and dash lookalikes read like a hyphen.
    joiner = pynini.union("​", "⁠")
    split_joiner = pynini.cdrewrite(pynini.cross(pynini.closure(joiner, 1), " "), any_digit, any_digit, NEMO_SIGMA)
    clean = (
        split_joiner
        @ pynini.cdrewrite(pynutil.delete(pynini.union(*_FORMAT_CHARS)), "", "", NEMO_SIGMA)
        @ pynini.cdrewrite(pynini.cross(pynini.union(*_DASHES), "-"), "", "", NEMO_SIGMA)
        @ pynini.cdrewrite(pynini.cross(pynini.union(*_SPACES), " "), "", "", NEMO_SIGMA)
    )

    # U+2212 MINUS SIGN between digits is subtraction; elsewhere it is a plain minus.
    minus = f" {MINUS_WORD} "
    true_minus = pynini.cdrewrite(
        pynini.cross("−", minus), any_digit + spaces, spaces + any_digit, NEMO_SIGMA
    ) @ pynini.cdrewrite(pynini.cross("−", "-"), "", "", NEMO_SIGMA)

    # %க்கு reads as a dative percent word; other case suffixes on % likewise
    # (data/whitelist/percent_suffix.tsv). Any other Tamil word glued to % is a separate word.
    trailing_punct = pynini.union(*[pynini.escape(c) for c in "()\"'{}[].,!?%"])
    percent_rows = [row for row in load_labels(get_abs_path("data/whitelist/percent_suffix.tsv")) if len(row) >= 2]
    percent_suffix = pynini.cdrewrite(
        pynini.union(*[pynini.cross(written, " " + spoken) for written, spoken, *_ in percent_rows]),
        any_digit,
        pynini.union(" ", "[EOS]", trailing_punct),
        NEMO_SIGMA,
    )
    percent_word = pynini.cdrewrite(pynutil.insert(" "), "%", letter, NEMO_SIGMA)

    # A hyphen inside an equation is a minus, not a range: 5-3=2, 10 - 5 = 5.
    subtraction_minus = pynini.cdrewrite(
        pynini.cross("-", minus),
        any_digit + spaces,
        spaces + pynini.closure(pynini.union(any_digit, "-", " "), 1) + "=",
        NEMO_SIGMA,
    )
    # <, > and + are markup or a sign except between two digits, where they are operators.
    comparison = pynini.cdrewrite(
        pynini.union(
            pynini.cross("<", f" {LESS_THAN} "),
            pynini.cross(">", f" {GREATER_THAN} "),
            pynini.cross("+", f" {INFIX_PLUS} "),
        ),
        any_digit + spaces,
        spaces + any_digit,
        NEMO_SIGMA,
    )

    # Split the symbols the whitelist speaks off digits and words: 5×3=15 -> 5 × 3 = 15, 5% -> 5 %.
    operator = pynini.union("×", "÷", "%", "=")
    space_after_digit = pynini.cdrewrite(pynutil.insert(" "), any_digit, operator, NEMO_SIGMA)
    space_before_digit = pynini.cdrewrite(pynutil.insert(" "), operator, any_digit, NEMO_SIGMA)
    spoken_symbol = pynini.union(*SPOKEN_SYMBOLS)
    split_symbol = pynini.cdrewrite(pynutil.insert(" "), NEMO_NOT_SPACE, spoken_symbol, NEMO_SIGMA) @ pynini.cdrewrite(
        pynutil.insert(" "), spoken_symbol, NEMO_NOT_SPACE, NEMO_SIGMA
    )
    # @ and _ are spoken too, but stay glued between ASCII letters or digits so an e-mail
    # address or an identifier (user@example.com, a_b) passes through whole.
    edge_symbol = pynini.union("@", "_")
    not_identifier = pynini.difference(NEMO_NOT_SPACE, pynini.union(NEMO_ALPHA, edge_symbol))
    split_edge_symbol = pynini.cdrewrite(
        pynutil.insert(" "), not_identifier, edge_symbol, NEMO_SIGMA
    ) @ pynini.cdrewrite(pynutil.insert(" "), edge_symbol, not_identifier, NEMO_SIGMA)
    # A hyphen between two amounts is a range: ₹5 - ₹10, ₹5-₹10.
    currency = pynini.union(*CURRENCY_SYMBOLS)
    money_range = pynini.cdrewrite(
        pynini.cross("-", f" {RANGE_WORD} "), any_digit + spaces, spaces + currency, NEMO_SIGMA
    )

    # A hyphen between a digit and a case/ordinal suffix belongs to the suffix (3-வது, 2024-ல்,
    # 100-க்கு); any other hyphen joining a digit to a Tamil word is a separator
    # (5-அவர்கள் -> 5 அவர்கள், 15-ஜூன்-2024 -> 15 ஜூன் 2024).
    drop_ordinal_hyphen = pynini.cdrewrite(pynutil.delete("-"), any_digit, known_suffixes, NEMO_SIGMA)
    joiner_hyphen_to_space = pynini.cdrewrite(
        pynini.cross("-", " "), any_digit, letter, NEMO_SIGMA
    ) @ pynini.cdrewrite(pynini.cross("-", " "), letter, any_digit, NEMO_SIGMA)
    # Case and ordinal suffixes may stay glued to a digit; anything else glued to a digit is a
    # separate word (5கிலோ -> 5 கிலோ).
    word = pynini.closure(letter, 1)
    unknown_word = pynini.difference(word, known_suffixes).optimize()
    boundary = pynini.union(" ", "[EOS]", pynini.difference(NEMO_CHAR, letter))
    split_digit_word = pynini.cdrewrite(pynutil.insert(" "), any_digit, unknown_word + boundary, NEMO_SIGMA)
    # A Tamil letter glued to a digit on its left is a separate word too (ஜி20, 5.மணி).
    letter_digit = pynini.cdrewrite(pynutil.insert(" "), letter, any_digit, NEMO_SIGMA)
    dot_letter = pynini.cdrewrite(pynutil.insert(" "), any_digit + ".", letter, NEMO_SIGMA)

    return (
        clean
        @ true_minus
        @ drop_ordinal_hyphen
        @ percent_suffix
        @ percent_word
        @ subtraction_minus
        @ comparison
        @ space_after_digit
        @ space_before_digit
        @ split_symbol
        @ split_edge_symbol
        @ money_range
        @ joiner_hyphen_to_space
        @ letter_digit
        @ dot_letter
        @ split_digit_word
    ).optimize()


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir,
                f"ta_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")

            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst

            fraction_graph = FractionFst(cardinal=cardinal, deterministic=deterministic).fst
            date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst
            time_graph = TimeFst(deterministic=deterministic).fst
            ordinal_graph = OrdinalFst(cardinal=cardinal, deterministic=deterministic).fst
            measure_graph = MeasureFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic).fst
            money_graph = MoneyFst(cardinal=cardinal, deterministic=deterministic).fst
            telephone_graph = TelephoneFst(cardinal=cardinal, deterministic=deterministic).fst
            range_graph = RangeFst(cardinal=cardinal, deterministic=deterministic).fst
            roman_graph = RomanFst(cardinal=cardinal, deterministic=deterministic).fst
            serial_graph = SerialFst(cardinal=cardinal, deterministic=deterministic).fst
            electronic_graph = ElectronicFst(deterministic=deterministic).fst

            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst

            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst

            word = WordFst(punctuation=punctuation, deterministic=deterministic)
            word_graph = word.fst

            # The number classes are ranked so that a span every one of them can read goes to
            # the most specific: a telephone shape before a cardinal, a date before a range, a
            # range (10-20) before two cardinals and a hyphen. Codes and addresses only exist
            # where no number class reads the span.
            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(telephone_graph, 0.5)
                | pynutil.add_weight(measure_graph, 1.03)
                | pynutil.add_weight(date_graph, 1.04)
                | pynutil.add_weight(time_graph, 1.05)
                | pynutil.add_weight(fraction_graph, 1.06)
                | pynutil.add_weight(decimal_graph, 1.08)
                | pynutil.add_weight(range_graph, 1.09)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(roman_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 1.1)
                | pynutil.add_weight(serial_graph, 1.2)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=2.1) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.union(
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space),
                    (pynutil.insert(NEMO_SPACE) + punct),
                ),
                1,
            )

            classify = pynini.union(classify, pynutil.add_weight(word_graph, 100))
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(NEMO_SPACE))
                + token
                + pynini.closure(pynutil.insert(NEMO_SPACE) + punct)
            )

            graph = token_plus_punct + pynini.closure(
                pynini.union(
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space),
                    (pynutil.insert(NEMO_SPACE) + punct + pynutil.insert(NEMO_SPACE)),
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph = pynini.union(graph, punct).optimize()

            # The spacing rewrites run over the text first, so every grammar above reads
            # clean digit runs and spaced symbols.
            self.fst = pynini.compose(_pre_process(cardinal.known_suffixes), graph).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
