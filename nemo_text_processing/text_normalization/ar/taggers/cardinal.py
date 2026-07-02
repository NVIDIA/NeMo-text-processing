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
        "9837" ->  cardinal { integer: "تسعة آلاف وثمان مئة وسبعة وثلاثين" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):

        super().__init__(name="cardinal", kind="classify")
        # zero
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        # cardinals data files
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv")).optimize()
        digit_100 = pynini.string_file(get_abs_path("data/number/digit_100.tsv")).optimize()
        digit_1000 = pynini.string_file(get_abs_path("data/number/digit_1000.tsv")).optimize()
        teens = pynini.string_file(get_abs_path("data/number/teens.tsv")).optimize()
        tens_nom = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/number/tens_nom.tsv")), weight=0.001
        ).optimize()
        tens_gen = pynini.string_file(get_abs_path("data/number/tens_gen.tsv")).optimize()

        # Grammar for cardinals 10_20_30 etc
        # add weight to prefer genetive case over nominative
        tens_zero_nom = tens_nom + pynutil.delete("0")
        tens_zero_nom = pynutil.add_weight(tens_zero_nom, weight=0.001)
        tens_zero_gen = tens_gen + pynutil.delete("0")

        # Creating flops for two digit cardinals 34->43
        reverse_digits = pynini.string_file(get_abs_path("data/number/flops.tsv"))

        # Grammar for two digitcardinals
        graph_flops = flop_digits @ reverse_digits
        # 34-- أربعة وثلاثون
        graph_tens_plus = graph_digit + insert_space + insert_and + (tens_nom | tens_gen)
        # flop
        graph_tens_plus_flop = graph_flops @ graph_tens_plus
        graph_all = graph_digit | teens | tens_zero_nom | graph_tens_plus_flop | tens_zero_gen
        graph_two_digits = teens | tens_zero_nom | graph_tens_plus_flop | tens_zero_gen

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
        two_hundreds = pynini.cross("2", "مئتين")
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

        # ---- counted-noun (تمييز) agreement for a 3-digit multiplier count (100-999) ----
        # The thousand/million word agrees with the *trailing* element of the count:
        #   trailing 3-10 -> plural (آلاف / ملايين),  otherwise -> singular (ألف / مليون).
        # e.g. 110 -> "مئة وعشرة آلاف", but 123 -> "مئة وثلاثة وعشرين ألف".
        _h = pynini.union("1", "2", "3", "4", "5", "6", "7", "8", "9")
        _digit_any = pynini.union("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
        # trailing value 3..10
        _trailing_plural = pynini.accep("0") + pynini.union("3", "4", "5", "6", "7", "8", "9") | pynini.accep("10")
        # trailing value 0,1,2 or 11..99
        _trailing_singular = (
            pynini.accep("0") + pynini.union("0", "1", "2")
            | pynini.accep("1") + pynini.union("1", "2", "3", "4", "5", "6", "7", "8", "9")
            | pynini.union("2", "3", "4", "5", "6", "7", "8", "9") + _digit_any
        )
        hundreds_count_plural = ((_h + _trailing_plural) @ graph_all_hundreds).optimize()
        hundreds_count_singular = ((_h + _trailing_singular) @ graph_all_hundreds).optimize()

        # ---- reusable building blocks (values 1-999) ----
        # a full 3-digit period (001-999) with internal leading zeros removed
        period_nonzero = (
            pynutil.delete("00") + graph_digit
            | pynutil.delete("0") + graph_two_digits
            | graph_all_hundreds
        )
        # trailing 3-digit remainder: either all zeros (nothing) or " و<words>"
        units_remainder = pynutil.delete("000") | (insert_space + insert_and + period_nonzero)

        # ---- thousands: 1_000 .. 999_999 ----
        # maps the thousand-count to "<count> <thousand-word>" with correct agreement:
        #   1 -> ألف, 2 -> ألفين, 3-10 -> <count> آلاف, 11-999 -> <count> ألف
        thousand_group = (
            pynini.cross("1", "ألف")
            | pynini.cross("2", "ألفين")
            | (digit_1000 + pynutil.insert(" آلاف"))
            | pynini.cross("10", "عشرة آلاف")
            | pynini.cross("200", "مئتي ألف")  # dual construct-state (drops nun before counted noun)
            | pynutil.add_weight(graph_two_digits + pynutil.insert(" ألف"), 0.01)
            | pynutil.add_weight(hundreds_count_plural + pynutil.insert(" آلاف"), 0.02)
            | pynutil.add_weight(hundreds_count_singular + pynutil.insert(" ألف"), 0.02)
        )
        graph_thousands = thousand_group + units_remainder

        # ---- millions: 1_000_000 .. 999_999_999 ----
        # same agreement pattern as thousands (مليون / مليونين / ملايين / مليون)
        million_group = (
            pynini.cross("1", "مليون")
            | pynini.cross("2", "مليونين")
            | (digit_1000 + pynutil.insert(" ملايين"))
            | pynini.cross("10", "عشرة ملايين")
            | pynini.cross("200", "مئتي مليون")  # dual construct-state (drops nun before counted noun)
            | pynutil.add_weight(graph_two_digits + pynutil.insert(" مليون"), 0.01)
            | pynutil.add_weight(hundreds_count_plural + pynutil.insert(" ملايين"), 0.02)
            | pynutil.add_weight(hundreds_count_singular + pynutil.insert(" مليون"), 0.02)
        )
        # zero-padded thousand-count (001-999) used inside a 6-digit remainder block
        thousand_group_padded = (
            pynini.cross("001", "ألف")
            | pynini.cross("002", "ألفين")
            | (pynutil.delete("00") + digit_1000 + pynutil.insert(" آلاف"))
            | pynini.cross("010", "عشرة آلاف")
            | pynini.cross("200", "مئتي ألف")  # dual construct-state (drops nun before counted noun)
            | pynutil.add_weight(pynutil.delete("0") + graph_two_digits + pynutil.insert(" ألف"), 0.01)
            | pynutil.add_weight(hundreds_count_plural + pynutil.insert(" آلاف"), 0.02)
            | pynutil.add_weight(hundreds_count_singular + pynutil.insert(" ألف"), 0.02)
        )
        # 6-digit remainder after the millions group (000001 .. 999999)
        block6_nonzero = (
            pynutil.delete("000") + period_nonzero
            | thousand_group_padded + units_remainder
        )
        million_remainder = pynutil.delete("000000") | (insert_space + insert_and + block6_nonzero)
        graph_millions = million_group + million_remainder

        self.graph = (
            graph_zero
            | graph_all
            | graph_all_hundreds
            | graph_thousands
            | graph_millions
        )

        #  remove leading zeros
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (leading_zeros + self.graph).optimize()

        self.cardinal_numbers = (self.graph | self.cardinal_numbers_with_leading_zeros).optimize()

        self.optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        final_graph = (
            self.optional_minus_graph
            + pynutil.insert('integer: "')
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)  # inserts the cardinal tag

        self.fst = final_graph
