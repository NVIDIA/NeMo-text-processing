# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
# Copyright 2015 and onwards Google, Inc.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.rw.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_CONSONANTS,
    NEMO_DIGIT,
    NEMO_VOWELS,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.rw.utils import get_abs_path


class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        vowels_or_space = NEMO_VOWELS | " "
        rewrite_na_fst = pynini.cdrewrite(
            pynini.cross(" ", " na "), vowels_or_space, NEMO_CONSONANTS, NEMO_CHAR.closure()
        )
        rewrite_n_fst = pynini.cdrewrite(pynini.cross(" ", " n'"), vowels_or_space, NEMO_VOWELS, NEMO_CHAR.closure())
        remove_underscore_fst = pynini.cdrewrite(
            pynini.cross("_", " "), pynini.union(NEMO_ALPHA), pynini.union(NEMO_ALPHA), NEMO_CHAR.closure()
        )
        remove_extra_space_fst = pynini.cdrewrite(
            delete_extra_space, pynini.union(NEMO_ALPHA), pynini.union(NEMO_ALPHA), NEMO_CHAR.closure()
        )
        remove_trailing_space_fst = pynini.cdrewrite(
            delete_space, pynini.union(NEMO_ALPHA).closure(), '[EOS]', NEMO_CHAR.closure()
        )

        rewrite_add_separator_fst = pynini.compose(rewrite_na_fst, rewrite_n_fst)
        ten_thousand = pynini.string_map([("ibihumbi_icumi", "10")])
        ten = pynini.string_map([("icumi", "10")])
        digits = pynini.string_file(get_abs_path("data/cardinal/digits.tsv"))
        digits_for_thousands = pynini.string_file(get_abs_path("data/cardinal/digits_for_thousands.tsv"))
        digits_millions_trillions = pynini.string_file(get_abs_path("data/cardinal/digits_millions_trillions.tsv"))
        tens = pynini.string_file(get_abs_path("data/cardinal/tens.tsv"))
        tens_for_ends = pynini.string_map([("icumi", "1")]) | tens
        tens_for_beginnings = pynini.string_map([("cumi", "1")]) | tens
        hundreds = pynini.string_file(get_abs_path("data/cardinal/hundreds.tsv"))
        thousands = pynini.string_file(get_abs_path("data/cardinal/thousands.tsv"))
        tens_of_thousands = pynini.string_file(get_abs_path("data/cardinal/tens_of_thousands.tsv"))
        hundreds_of_thousands = pynini.string_file(get_abs_path("data/cardinal/hundreds_of_thousands.tsv"))
        millions = pynini.string_file(get_abs_path("data/cardinal/millions.tsv"))
        tens_of_millions = pynini.string_file(get_abs_path("data/cardinal/tens_of_millions.tsv"))
        hundreds_of_millions = pynini.string_file(get_abs_path("data/cardinal/hundreds_of_millions.tsv"))
        trillions = pynini.string_file(get_abs_path("data/cardinal/trillions.tsv"))
        tens_of_trillions = pynini.string_file(get_abs_path("data/cardinal/tens_of_trillions.tsv"))
        hundreds_of_trillions = pynini.string_file(get_abs_path("data/cardinal/hundreds_of_trillions.tsv"))

        THREE_ZEROS = "000"
        FOUR_ZEROS = "0000"
        FIVE_ZEROS = "00000"
        SIX_ZEROS = "000000"
        SEVEN_ZEROS = "0000000"
        EIGHT_ZEROS = "00000000"
        NINE_ZEROS = "000000000"

        zero = pynini.string_map([("zeru", "0")])
        rewrite_remove_comma_fst = pynini.cdrewrite(
            pynini.cross(",", ""), pynini.union(NEMO_DIGIT), pynini.union(NEMO_DIGIT), NEMO_CHAR.closure()
        )
        single_digits_graph = pynini.invert(digits | zero)
        single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)
        remove_comma = rewrite_remove_comma_fst @ single_digits_graph

        graph_tens_ends = tens_for_ends + pynutil.delete(" ") + digits | tens_for_ends + pynutil.insert("0")
        graph_tens_starts = tens_for_beginnings + pynutil.delete(" ") + digits | tens_for_beginnings + pynutil.insert(
            "0"
        )

        graph_tens_for_thousands = tens_for_beginnings + pynutil.delete(
            " "
        ) + digits_for_thousands | tens_for_beginnings + pynutil.insert("0")

        graph_tens_for_millions_trillions = tens_for_beginnings + pynutil.delete(
            " "
        ) + digits_millions_trillions | tens_for_beginnings + pynutil.insert("0")
        graph_hundreds = (
            hundreds + pynutil.delete(" ") + graph_tens_ends
            | hundreds + pynutil.insert("00")
            | hundreds + pynutil.delete(" ") + pynutil.insert("0") + digits
        )
        graph_thousands = (
            thousands + pynutil.delete(" ") + graph_hundreds
            | thousands + pynutil.insert(THREE_ZEROS)
            | thousands + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_ends
            | thousands + pynutil.delete(" ") + pynutil.insert("00") + digits
        )

        graph_ten_thousand_and_hundreds = (
            ten_thousand + pynutil.insert(THREE_ZEROS)
            | ten_thousand + pynutil.delete(" ") + graph_hundreds
            | ten_thousand + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_ends
            | ten_thousand + pynutil.delete(" ") + pynutil.insert("00") + digits
        )
        prefix_tens_of_thousands = tens_of_thousands + pynutil.delete(" ") + digits_for_thousands
        graph_tens_of_thousands = (
            pynutil.add_weight(graph_ten_thousand_and_hundreds, weight=-0.1)
            | prefix_tens_of_thousands + pynutil.delete(" ") + graph_hundreds
            | prefix_tens_of_thousands + pynutil.insert(THREE_ZEROS)
            | prefix_tens_of_thousands + pynutil.delete(" ") + pynutil.insert("0") + graph_hundreds
            | prefix_tens_of_thousands + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_ends
            | prefix_tens_of_thousands + pynutil.delete(" ") + pynutil.insert("00") + digits
        )

        prefix_hundreds_of_thousands = hundreds_of_thousands + pynutil.delete(" ") + graph_tens_for_thousands
        graph_hundreds_of_thousands = (
            hundreds_of_thousands + pynutil.insert(FIVE_ZEROS)
            | prefix_hundreds_of_thousands + pynutil.insert(THREE_ZEROS)
            | prefix_hundreds_of_thousands + pynutil.delete(" ") + graph_hundreds
            | pynutil.add_weight(
                prefix_hundreds_of_thousands + pynutil.delete(" ") + pynutil.insert("00") + digits, weight=-0.1
            )
            | prefix_hundreds_of_thousands + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_for_thousands
        )

        graph_millions = (
            millions + pynutil.delete(" ") + graph_hundreds_of_thousands
            | millions + pynutil.insert(SIX_ZEROS)
            | millions + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_of_thousands
            | millions + pynutil.delete(" ") + pynutil.insert("00") + graph_thousands
            | millions + pynutil.delete(" ") + pynutil.insert(THREE_ZEROS) + graph_hundreds
            | millions + pynutil.delete(" ") + pynutil.insert(FOUR_ZEROS) + graph_tens_ends
            | millions + pynutil.delete(" ") + pynutil.insert(FIVE_ZEROS) + digits
        )

        prefix_tens_of_millions = tens_of_millions + pynutil.delete(" ") + digits_millions_trillions
        graph_tens_of_millions = (
            prefix_tens_of_millions + pynutil.delete(" ") + graph_hundreds_of_thousands
            | prefix_tens_of_millions + pynutil.delete(" ") + pynutil.insert(SIX_ZEROS)
            | prefix_tens_of_millions + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_of_thousands
            | prefix_tens_of_millions + pynutil.delete(" ") + pynutil.insert(THREE_ZEROS) + graph_hundreds
            | prefix_tens_of_millions + pynutil.delete(" ") + pynutil.insert(FOUR_ZEROS) + graph_tens_ends
            | tens_of_millions + pynutil.delete(" ") + pynutil.insert(FIVE_ZEROS) + graph_tens_ends
            | prefix_tens_of_millions + pynutil.delete(" ") + pynutil.insert(FIVE_ZEROS) + digits
        )

        prefix_hundreds_of_millions = hundreds_of_millions + pynutil.delete(" ") + graph_tens_for_millions_trillions
        graph_hundreds_of_millions = (
            prefix_hundreds_of_millions + pynutil.delete(" ") + graph_hundreds_of_thousands
            | prefix_hundreds_of_millions + pynutil.insert(SIX_ZEROS)
            | prefix_hundreds_of_millions + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_of_thousands
            | prefix_hundreds_of_millions + pynutil.delete(" ") + pynutil.insert("00") + graph_thousands
            | prefix_hundreds_of_millions + pynutil.delete(" ") + pynutil.insert(THREE_ZEROS) + graph_hundreds
            | prefix_hundreds_of_millions + pynutil.delete(" ") + pynutil.insert(FOUR_ZEROS) + graph_tens_ends
        )

        graph_trillions = (
            trillions + pynutil.delete(" ") + graph_hundreds_of_millions
            | trillions + pynutil.insert(NINE_ZEROS)
            | trillions + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_of_millions
            | trillions + pynutil.delete(" ") + pynutil.insert("00") + graph_millions
            | trillions + pynutil.delete(" ") + pynutil.insert(THREE_ZEROS) + graph_hundreds_of_thousands
            | trillions + pynutil.delete(" ") + pynutil.insert(FOUR_ZEROS) + graph_tens_of_thousands
            | trillions + pynutil.delete(" ") + pynutil.insert(FIVE_ZEROS) + graph_thousands
            | trillions + pynutil.delete(" ") + pynutil.insert(SIX_ZEROS) + graph_hundreds
            | trillions + pynutil.delete(" ") + pynutil.insert(SEVEN_ZEROS) + graph_tens_ends
            | trillions + pynutil.delete(" ") + pynutil.insert(EIGHT_ZEROS) + digits
        )

        prefix_tens_of_trillions = tens_of_trillions + pynutil.delete(" ") + digits_millions_trillions
        graph_tens_of_trillions = (
            prefix_tens_of_trillions + pynutil.delete(" ") + graph_hundreds_of_millions
            | prefix_tens_of_trillions + pynutil.insert(NINE_ZEROS)
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_of_millions
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert("00") + graph_millions
            | prefix_tens_of_trillions
            + pynutil.delete(" ")
            + pynutil.insert(THREE_ZEROS)
            + graph_hundreds_of_thousands
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert(FOUR_ZEROS) + graph_tens_of_thousands
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert(FIVE_ZEROS) + graph_thousands
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert(SIX_ZEROS) + graph_hundreds
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert(SEVEN_ZEROS) + graph_tens_ends
            | prefix_tens_of_trillions + pynutil.delete(" ") + pynutil.insert(EIGHT_ZEROS) + digits
        )

        prefix_hundreds_of_trillions = hundreds_of_trillions + pynutil.delete(" ") + graph_tens_for_millions_trillions
        graph_hundreds_of_trillions = (
            prefix_hundreds_of_trillions + pynutil.delete(" ") + graph_hundreds_of_millions
            | prefix_hundreds_of_trillions + pynutil.insert(NINE_ZEROS)
            | prefix_hundreds_of_trillions + pynutil.delete(" ") + pynutil.insert("0") + graph_tens_of_millions
            | prefix_hundreds_of_trillions + pynutil.delete(" ") + pynutil.insert("00") + graph_millions
            | prefix_hundreds_of_trillions
            + pynutil.delete(" ")
            + pynutil.insert(THREE_ZEROS)
            + graph_hundreds_of_thousands
            | prefix_hundreds_of_trillions + pynutil.delete(" ") + pynutil.insert(FOUR_ZEROS) + graph_tens_of_thousands
            | prefix_hundreds_of_trillions + pynutil.delete(" ") + pynutil.insert(FIVE_ZEROS) + graph_thousands
            | prefix_hundreds_of_trillions + pynutil.delete(" ") + pynutil.insert(SIX_ZEROS) + graph_hundreds
            | prefix_hundreds_of_trillions + pynutil.delete(" ") + pynutil.insert(SEVEN_ZEROS) + graph_tens_ends
        )

        graph_all = (
            graph_hundreds_of_trillions
            | graph_tens_of_trillions
            | graph_trillions
            | graph_hundreds_of_millions
            | graph_tens_of_millions
            | graph_millions
            | graph_hundreds_of_thousands
            | graph_tens_of_thousands
            | graph_thousands
            | graph_hundreds
            | pynutil.add_weight(ten, weight=-0.1)
            | graph_tens_starts
            | digits
            | pynini.cross("zeru", "0")
        )

        inverted_graph_all = pynini.compose(pynini.invert(graph_all), rewrite_add_separator_fst)
        inverted_graph_all = pynini.compose(inverted_graph_all, remove_extra_space_fst)
        inverted_graph_all = pynini.compose(inverted_graph_all, remove_trailing_space_fst)
        inverted_graph_all = pynini.compose(inverted_graph_all, remove_underscore_fst) | pynutil.add_weight(
            remove_comma, 0.0001
        )

        inverted_graph_all = inverted_graph_all.optimize()
        final_graph = pynutil.insert("integer: \"") + inverted_graph_all + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
