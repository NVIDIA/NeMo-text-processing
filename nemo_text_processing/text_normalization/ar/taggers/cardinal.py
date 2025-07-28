# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import GraphFst, flop_digits, insert_and, insert_space
from nemo_text_processing.text_normalization.ar.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "9837" ->  cardinal { integer: "تسعة اَلاف وثمان مئة وسبعة وثلاثون" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):

        super().__init__(name="cardinal", kind="classify")
        # zero
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        # cardinals data files
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        digit_100 = pynini.string_file(get_abs_path("data/number/digit_100.tsv"))
        digit_1000 = pynini.string_file(get_abs_path("data/number/digit_1000.tsv"))
        teens = pynini.string_file(get_abs_path("data/number/teens.tsv"))
        tens = pynini.string_file(get_abs_path("data/number/tens.tsv"))

        # Grammar for cardinals 10_20_30 etc
        tens_zero = tens + pynutil.delete("0")

        # Creating flops for two digit cardinals 34->43
        reverse_digits = pynini.string_file(get_abs_path("data/number/flops.tsv"))

        # Grammar for two digitcardinals
        graph_flops = flop_digits @ reverse_digits
        graph_tens_plus1 = graph_digit + insert_space + insert_and + tens
        graph_tens_plus = graph_flops @ graph_tens_plus1
        graph_all = graph_digit | teens | tens_zero | graph_tens_plus
        graph_two_digits = teens | tens_zero | graph_tens_plus

        # Grammar for cardinals hundreds
        one_hundred = pynini.cross("1", "مئة")
        hundreds_zero = digit_100 + insert_space + pynutil.insert("مئة") + pynutil.delete("00", weight=0.001)
        hundreds_plus = (
            digit_100 + insert_space + pynutil.insert("مئة") + insert_space + insert_and + graph_two_digits
            | digit_100
            + insert_space
            + pynutil.insert("مئة")
            + pynutil.delete("0")
            + insert_space
            + insert_and
            + graph_digit
        )
        two_hundreds = pynini.cross("2", "مئتان")
        graph_one_hundred = one_hundred + pynutil.delete("00", weight=0.001)
        graph_one_hundred_plus = (
            one_hundred + insert_space + insert_and + graph_two_digits
            | one_hundred + pynutil.delete("0") + insert_space + insert_and + graph_digit
        )
        graph_two_hundreds = (
            (two_hundreds + pynutil.delete("00", weight=0.001))
            | two_hundreds + insert_space + insert_and + graph_two_digits
            | two_hundreds + pynutil.delete("0") + insert_space + insert_and + graph_digit
        )

        graph_all_one_hundred = graph_one_hundred | graph_one_hundred_plus

        graph_all_hundreds = graph_all_one_hundred | graph_two_hundreds | hundreds_zero | hundreds_plus

        # Grammar for thousands
        one_thousand = pynini.cross("1", "ألف")
        thousands_zero = digit_1000 + insert_space + pynutil.insert("اَلاف") + pynutil.delete("000", weight=0.001)
        thousands_plus = (
            digit_1000 + insert_space + pynutil.insert("اَلاف") + insert_space + insert_and + graph_all_hundreds
        )
        thousands_skip_hundreds = (
            digit_1000
            + insert_space
            + pynutil.insert("اَلاف")
            + pynutil.delete("0")
            + insert_space
            + insert_and
            + graph_two_digits
            | digit_1000
            + insert_space
            + pynutil.insert("اَلاف")
            + pynutil.delete("00")
            + insert_space
            + insert_and
            + graph_digit
        )
        two_thousands = pynini.cross("2", "ألفان")
        graph_one_thousand = one_thousand + pynutil.delete("000", weight=0.001)
        graph_one_thousand_plus = (
            one_thousand + insert_space + insert_and + graph_all_hundreds
            | one_thousand + pynutil.delete("0") + insert_space + insert_and + graph_two_digits
            | one_thousand + pynutil.delete("00") + insert_space + insert_and + graph_digit
        )
        graph_two_thousands = two_thousands + pynutil.delete("000", weight=0.001)
        graph_two_thousands_plus = (
            two_thousands + insert_space + insert_and + graph_all_hundreds
            | two_thousands + pynutil.delete("0") + insert_space + insert_and + graph_two_digits
            | two_thousands + pynutil.delete("00") + insert_space + insert_and + graph_digit
        )

        graph_all_one_thousand = graph_one_thousand | graph_one_thousand_plus
        graph_all_two_thousands = graph_two_thousands | graph_two_thousands_plus

        graph_all_thousands = (
            graph_all_one_thousand
            | graph_two_thousands
            | thousands_zero
            | thousands_plus
            | thousands_skip_hundreds
            | graph_all_two_thousands
        )

        graph = graph_all | graph_all_hundreds | graph_all_thousands | graph_zero

        self.cardinal_numbers = (graph).optimize()

        #  remove leading zeros
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (leading_zeros + self.cardinal_numbers).optimize()

        self.optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        final_graph = (
            self.optional_minus_graph
            + pynutil.insert('integer: "')
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)  # inserts the cardinal tag

        self.fst = final_graph
