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

import re

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hu.graph_utils import ensure_space
from nemo_text_processing.text_normalization.hu.utils import (
    get_abs_path,
    inflect_abbreviation,
    load_labels,
    naive_inflector,
)

min_singular = pynini.string_file(get_abs_path("data/money/currency_minor.tsv"))
maj_singular = pynini.string_file((get_abs_path("data/money/currency.tsv")))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "€1" -> money { currency_maj: "euró" integer_part: "egy"}
        "€1,000" -> money { currency_maj: "euró" integer_part: "egy" }
        "€1,001" -> money { currency_maj: "euró" integer_part: "egy" fractional_part: "egy"}
        "£1,4" -> money { integer_part: "egy" currency_maj: "font" fractional_part: "negyven" preserve_order: true}
               -> money { integer_part: "egy" currency_maj: "font" fractional_part: "negyven" currency_min: "penny" preserve_order: true}
        "£0,01" -> money { fractional_part: "egy" currency_min: "penny" preserve_order: true}
        "£0,01 million" -> money { currency_maj: "font" integer_part: "nulla" fractional_part: "egy század" quantity: "millió"}

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.final_graph_wo_negative

        maj_singular_labels = load_labels(get_abs_path("data/money/currency.tsv"))
        maj_singular_graph = convert_space(maj_singular)

        letters = pynini.string_file((get_abs_path("data/money/alphabet.tsv")))
        if not deterministic:
            letters |= pynini.cross("W", "vé")
            letters |= pynini.cross("W", "kettős vé")
        read_letters = letters + pynini.closure(insert_space + letters)
        self.read_letters = read_letters

        graph_maj_singular = pynutil.insert("currency_maj: \"") + maj_singular_graph + pynutil.insert("\"")

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(",") + pynini.closure(pynutil.delete("0"), 1), 0, 1
        )

        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(NEMO_DIGIT, 1)
            + pynini.accep(",")
            + pynini.closure(NEMO_DIGIT, 2)
            + (NEMO_DIGIT - "0")
            + pynini.closure(pynutil.delete("0"))
        )
        decimal_with_quantity = NEMO_SIGMA + NEMO_ALPHA
        decimal_part = (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final
        graph_decimal = graph_maj_singular + insert_space + decimal_part

        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        graph_integer_only = graph_maj_singular + insert_space + graph_integer

        graph = (graph_integer_only + optional_delete_fractional_zeros) | graph_decimal

        # remove trailing zeros of non zero number in the first 2 digits and fill up to 2 digits
        # e.g. 2000 -> 20, 0200->02, 01 -> 01, 10 -> 10
        # not accepted: 002, 00, 0,
        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )

        graph_min_singular = pynutil.insert(" currency_min: \"") + min_singular + pynutil.insert("\"")

        # format ** euro ** cent
        decimal_graph_with_minor = None
        for curr_symbol, cur_word in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: true")
            integer_plus_maj = graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular
            # non zero integer part
            integer_plus_maj = (pynini.closure(NEMO_DIGIT) - "0") @ integer_plus_maj

            abbr_expansion = pynini.string_map(naive_inflector(curr_symbol, cur_word))
            maj_inflected = pynini.accep(cur_word)
            maj_inflected |= pynini.project(abbr_expansion, "output")
            # where a currency abbreviation (like GBP) appears inflected (GBP-t),
            # we read the number as a pure fraction, because to add a minor currency
            # would involve moving the inflectional piece from major to minor
            if re.match("^[A-Z]{3}$", curr_symbol):
                letter_expansion = pynini.string_map(inflect_abbreviation(curr_symbol, cur_word))
                maj_inflected = letter_expansion | abbr_expansion
                maj_inflected |= pynini.cross(curr_symbol, cur_word)
                if not deterministic:
                    expanded = curr_symbol @ read_letters
                    get_endings = pynini.project(letter_expansion, "input")
                    letter_endings = get_endings @ (
                        pynini.cdrewrite(pynini.cross(f"{curr_symbol}-", expanded), "[BOS]", "", NEMO_SIGMA)
                    )
                    maj_inflected |= letter_endings
                    maj_inflected |= pynini.project(letter_endings, "output")
            graph_maj_final = pynutil.insert("currency_maj: \"") + maj_inflected + pynutil.insert("\"")
            graph |= graph_decimal_final + ensure_space + graph_maj_final + preserve_order
            graph |= graph_integer + ensure_space + graph_maj_final + preserve_order

            graph_fractional = (
                two_digits_fractional_part @ pynini.closure(NEMO_DIGIT, 1, 2) @ cardinal.two_digit_non_zero
            )
            graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional + pynutil.insert("\"")

            fractional_plus_min = graph_fractional + pynutil.insert(curr_symbol) @ graph_min_singular

            decimal_graph_with_minor_curr = integer_plus_maj + pynini.cross(",", " ") + fractional_plus_min

            decimal_graph_with_minor_curr |= pynutil.delete("0,") + fractional_plus_min
            decimal_graph_with_minor_curr = (
                pynutil.delete(curr_symbol) + decimal_graph_with_minor_curr + preserve_order
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        final_graph = graph | decimal_graph_with_minor

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
