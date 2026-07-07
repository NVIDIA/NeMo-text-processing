# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License.import pynini
import pynini
from pynini.lib import pynutil

import nemo_text_processing.inverse_text_normalization.ta.utils
from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst, delete_space


class CardinalFst(GraphFst):

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        graph_zero = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/zero.tsv")
        ).invert()

        graph_digit = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/digit.tsv")
        ).invert()

        graph_teens_and_ties = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/teens_and_ties.tsv")
        ).invert()

        graph_hundreds = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/hundreds.tsv")
        ).invert()

        graph_hundreds_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/hundreds_join.tsv")
        ).invert()

        graph_thousands = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/thousands.tsv")
        ).invert()

        graph_thousands_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/thousands_join.tsv")
        ).invert()

        graph_thousands_10_99 = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/thousands_10_99.tsv")
        ).invert()

        graph_thousands_10_99_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path(
                "data/numbers/thousands_10_99_join.tsv"
            )
        ).invert()

        graph_lakhs = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/lakhs.tsv")
        ).invert()

        graph_lakhs_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/lakhs_join.tsv")
        ).invert()

        graph_lakhs_10_99 = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/lakhs_10_99.tsv")
        ).invert()

        graph_lakhs_10_99_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/lakhs_10_99_join.tsv")
        ).invert()

        graph_crores = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/crores.tsv")
        ).invert()

        graph_crores_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/crores_join.tsv")
        ).invert()

        graph_crores_10_99 = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/crores_10_99.tsv")
        ).invert()

        graph_crores_10_99_join = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/crores_10_99_join.tsv")
        ).invert()

        case_graph = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/case_suffix.tsv")
        )

        self.graph_single_digit = graph_digit

        graph_digit_with_zero = (
            pynini.cross("ஒன்று", "௦௧")
            | pynini.cross("இரண்டு", "௦௨")
            | pynini.cross("மூன்று", "௦௩")
            | pynini.cross("நான்கு", "௦௪")
            | pynini.cross("ஐந்து", "௦௫")
            | pynini.cross("ஆறு", "௦௬")
            | pynini.cross("ஏழு", "௦௭")
            | pynini.cross("எட்டு", "௦௮")
            | pynini.cross("ஒன்பது", "௦௯")
        )

        # 10-99
        self.graph_two_digit = graph_teens_and_ties

        # 100, 200, ... 900
        self.graph_exact_hundreds = graph_hundreds + pynutil.insert("௦௦")

        # 101-109
        graph_hundred_digit = graph_hundreds_join + delete_space + graph_digit_with_zero

        # 110-199, 120-199, ...
        graph_hundred_two_digit = graph_hundreds_join + delete_space + graph_teens_and_ties

        self.graph_hundred_with_remainder = graph_hundred_digit | graph_hundred_two_digit

        # 1000, 2000 ... 9000
        self.graph_exact_thousands = graph_thousands + pynutil.insert("௦௦௦")

        graph_thousand_digit = graph_thousands_join + pynutil.insert("௦௦") + delete_space + self.graph_single_digit

        # 1010-1099
        graph_thousand_two_digit = graph_thousands_join + pynutil.insert("௦") + delete_space + self.graph_two_digit

        # 1100, 1200 ... 1900
        graph_thousand_hundred = graph_thousands_join + delete_space + self.graph_exact_hundreds

        # 1101-1999 ... 9901-9999
        graph_thousand_hundred_remainder = graph_thousands_join + delete_space + self.graph_hundred_with_remainder

        self.graph_thousand_with_remainder = (
            graph_thousand_digit | graph_thousand_two_digit | graph_thousand_hundred | graph_thousand_hundred_remainder
        )

        # 10000 - 99000
        # =========================
        graph_exact_large_thousands = graph_thousands_10_99 + pynutil.insert("௦௦௦")

        graph_large_thousand_digit = (
            graph_thousands_10_99_join + pynutil.insert("௦௦") + delete_space + self.graph_single_digit
        )

        graph_large_thousand_two_digit = (
            graph_thousands_10_99_join + pynutil.insert("௦") + delete_space + self.graph_two_digit
        )

        graph_large_thousand_hundred = graph_thousands_10_99_join + delete_space + self.graph_exact_hundreds

        graph_large_thousand_hundred_remainder = (
            graph_thousands_10_99_join + delete_space + self.graph_hundred_with_remainder
        )

        self.graph_large_thousands = (
            graph_exact_large_thousands
            | graph_large_thousand_digit
            | graph_large_thousand_two_digit
            | graph_large_thousand_hundred
            | graph_large_thousand_hundred_remainder
        )

        # 1,00,000 - 9,00,000
        graph_exact_lakhs = graph_lakhs + pynutil.insert("௦௦௦௦௦")

        # 10,00,000 - 99,00,000
        graph_exact_lakhs_10_99 = graph_lakhs_10_99 + pynutil.insert("௦௦௦௦௦")

        graph_lakh_digit = graph_lakhs_join + pynutil.insert("௦௦௦௦") + delete_space + self.graph_single_digit
        graph_lakh_two_digit = graph_lakhs_join + pynutil.insert("௦௦௦") + delete_space + self.graph_two_digit
        graph_lakh_hundred = graph_lakhs_join + pynutil.insert("௦௦") + delete_space + self.graph_exact_hundreds
        graph_lakh_hundred_remainder = (
            graph_lakhs_join + pynutil.insert("௦௦") + delete_space + self.graph_hundred_with_remainder
        )

        graph_lakh_thousand_remainder = (
            graph_lakhs_join + pynutil.insert("௦") + delete_space + self.graph_thousand_with_remainder
        )
        graph_lakh_large_thousand = graph_lakhs_join + delete_space + graph_exact_large_thousands
        graph_lakh_large_thousand_remainder = graph_lakhs_join + delete_space + self.graph_large_thousands

        graph_lakh_thousand = graph_lakhs_join + pynutil.insert("௦") + delete_space + self.graph_exact_thousands

        graph_lakh_remainder = (
            graph_lakh_digit
            | graph_lakh_two_digit
            | graph_lakh_hundred
            | graph_lakh_hundred_remainder
            | graph_lakh_thousand
            | graph_lakh_thousand_remainder
            | graph_lakh_large_thousand
            | graph_lakh_large_thousand_remainder
        )

        graph_lakh_10_99_digit = (
            graph_lakhs_10_99_join + pynutil.insert("௦௦௦௦") + delete_space + self.graph_single_digit
        )

        graph_lakh_10_99_two_digit = (
            graph_lakhs_10_99_join + pynutil.insert("௦௦௦") + delete_space + self.graph_two_digit
        )

        graph_lakh_10_99_hundred = (
            graph_lakhs_10_99_join + pynutil.insert("௦௦") + delete_space + self.graph_exact_hundreds
        )

        graph_lakh_10_99_hundred_remainder = (
            graph_lakhs_10_99_join + pynutil.insert("௦௦") + delete_space + self.graph_hundred_with_remainder
        )

        graph_lakh_10_99_thousand = (
            graph_lakhs_10_99_join + pynutil.insert("௦") + delete_space + self.graph_exact_thousands
        )

        graph_lakh_10_99_thousand_remainder = (
            graph_lakhs_10_99_join + pynutil.insert("௦") + delete_space + self.graph_thousand_with_remainder
        )

        graph_lakh_10_99_large_thousand = graph_lakhs_10_99_join + delete_space + graph_exact_large_thousands

        graph_lakh_10_99_large_thousand_remainder = graph_lakhs_10_99_join + delete_space + self.graph_large_thousands

        graph_lakh_10_99_remainder = (
            graph_lakh_10_99_digit
            | graph_lakh_10_99_two_digit
            | graph_lakh_10_99_hundred
            | graph_lakh_10_99_hundred_remainder
            | graph_lakh_10_99_thousand
            | graph_lakh_10_99_thousand_remainder
            | graph_lakh_10_99_large_thousand
            | graph_lakh_10_99_large_thousand_remainder
        )

        self.graph_lakh_numbers = (
            graph_exact_lakhs | graph_lakh_remainder | graph_exact_lakhs_10_99 | graph_lakh_10_99_remainder
        )

        # =====================================================
        # CRORES (1-9)

        graph_exact_crores = graph_crores + pynutil.insert("௦௦௦௦௦௦௦")

        graph_crore_digit = graph_crores_join + pynutil.insert("௦௦௦௦௦௦") + delete_space + self.graph_single_digit

        graph_crore_two_digit = graph_crores_join + pynutil.insert("௦௦௦௦௦") + delete_space + self.graph_two_digit

        graph_crore_hundred = graph_crores_join + pynutil.insert("௦௦௦௦") + delete_space + self.graph_exact_hundreds

        graph_crore_hundred_remainder = (
            graph_crores_join + pynutil.insert("௦௦௦௦") + delete_space + self.graph_hundred_with_remainder
        )

        graph_crore_thousand = graph_crores_join + pynutil.insert("௦௦௦") + delete_space + self.graph_exact_thousands

        graph_crore_thousand_remainder = (
            graph_crores_join + pynutil.insert("௦௦௦") + delete_space + self.graph_thousand_with_remainder
        )

        graph_crore_large_thousand = (
            graph_crores_join + pynutil.insert("௦௦") + delete_space + graph_exact_large_thousands
        )

        graph_crore_large_thousand_remainder = (
            graph_crores_join + pynutil.insert("௦௦") + delete_space + self.graph_large_thousands
        )

        graph_crore_lakh_10_99 = graph_crores_join + delete_space + graph_exact_lakhs_10_99

        graph_crore_lakh_10_99_remainder = graph_crores_join + delete_space + graph_lakh_10_99_remainder

        graph_crore_lakh = graph_crores_join + pynutil.insert("௦") + delete_space + graph_exact_lakhs

        graph_crore_lakh_remainder = graph_crores_join + pynutil.insert("௦") + delete_space + graph_lakh_remainder

        graph_crore_remainder = (
            graph_crore_digit
            | graph_crore_two_digit
            | graph_crore_hundred
            | graph_crore_hundred_remainder
            | graph_crore_thousand
            | graph_crore_thousand_remainder
            | graph_crore_large_thousand
            | graph_crore_large_thousand_remainder
            | graph_crore_lakh
            | graph_crore_lakh_remainder
            | graph_crore_lakh_10_99
            | graph_crore_lakh_10_99_remainder
        )

        # =====================================================
        # CRORES (10-99)

        graph_exact_crores_10_99 = graph_crores_10_99 + pynutil.insert("௦௦௦௦௦௦௦")

        graph_crore_10_99_remainder = (
            graph_crores_10_99_join
            + delete_space
            + (
                self.graph_single_digit
                | self.graph_two_digit
                | self.graph_exact_hundreds
                | self.graph_hundred_with_remainder
                | self.graph_exact_thousands
                | self.graph_thousand_with_remainder
                | graph_exact_large_thousands
                | self.graph_large_thousands
                | graph_exact_lakhs
                | graph_lakh_remainder
                | graph_exact_lakhs_10_99
                | graph_lakh_10_99_remainder
            )
        )

        self.graph_crore_numbers = (
            graph_exact_crores | graph_crore_remainder | graph_exact_crores_10_99 | graph_crore_10_99_remainder
        )

        # CRORE MULTIPLIERS (>99 CRORES)
        # =====================================================

        graph_crore_multiplier_base = (
            self.graph_exact_hundreds
            | self.graph_hundred_with_remainder
            | self.graph_exact_thousands
            | self.graph_thousand_with_remainder
            | graph_exact_large_thousands
            | self.graph_large_thousands
            | self.graph_lakh_numbers
        )

        graph_exact_crore_multiplier = (
            graph_crore_multiplier_base + delete_space + pynini.cross("கோடி", "") + pynutil.insert("௦௦௦௦௦௦௦")
        )

        graph_crore_multiplier_digit = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦௦௦௦௦௦")
            + delete_space
            + self.graph_single_digit
        )

        graph_crore_multiplier_two_digit = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦௦௦௦௦")
            + delete_space
            + self.graph_two_digit
        )

        graph_crore_multiplier_hundred = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦௦௦௦")
            + delete_space
            + self.graph_exact_hundreds
        )

        graph_crore_multiplier_hundred_remainder = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦௦௦௦")
            + delete_space
            + self.graph_hundred_with_remainder
        )

        graph_crore_multiplier_thousand = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦௦௦")
            + delete_space
            + self.graph_exact_thousands
        )

        graph_crore_multiplier_thousand_remainder = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦௦௦")
            + delete_space
            + self.graph_thousand_with_remainder
        )

        graph_crore_multiplier_large_thousand = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦")
            + delete_space
            + graph_exact_large_thousands
        )

        graph_crore_multiplier_large_thousand_remainder = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + pynutil.insert("௦")
            + delete_space
            + self.graph_large_thousands
        )

        graph_crore_multiplier_lakh = (
            graph_crore_multiplier_base + delete_space + pynini.cross("கோடியே", "") + delete_space + graph_exact_lakhs
        )

        graph_crore_multiplier_lakh_remainder = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + delete_space
            + graph_lakh_remainder
        )

        graph_crore_multiplier_lakh_10_99 = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + delete_space
            + graph_exact_lakhs_10_99
        )

        graph_crore_multiplier_lakh_10_99_remainder = (
            graph_crore_multiplier_base
            + delete_space
            + pynini.cross("கோடியே", "")
            + delete_space
            + graph_lakh_10_99_remainder
        )

        graph_crore_multiplier_remainder = (
            graph_crore_multiplier_digit
            | graph_crore_multiplier_two_digit
            | graph_crore_multiplier_hundred
            | graph_crore_multiplier_hundred_remainder
            | graph_crore_multiplier_thousand
            | graph_crore_multiplier_thousand_remainder
            | graph_crore_multiplier_large_thousand
            | graph_crore_multiplier_large_thousand_remainder
            | graph_crore_multiplier_lakh
            | graph_crore_multiplier_lakh_remainder
            | graph_crore_multiplier_lakh_10_99
            | graph_crore_multiplier_lakh_10_99_remainder
        )

        self.graph_large_crore_numbers = graph_exact_crore_multiplier | graph_crore_multiplier_remainder

        graph = (
            graph_zero
            | self.graph_single_digit
            | self.graph_two_digit
            | self.graph_exact_hundreds
            | self.graph_hundred_with_remainder
            | self.graph_exact_thousands
            | self.graph_thousand_with_remainder
            | self.graph_large_thousands
            | self.graph_lakh_numbers
            | self.graph_crore_numbers
            | self.graph_large_crore_numbers
        )

        self.graph_no_exception = graph.optimize()

        suffix_graph = pynini.string_map(
            [
                ("ஒன்பதின்", "௯ இன்"),
                ("பத்தின்", "௧௦ இன்"),
                ("பத்தில்", "௧௦ இல்"),
                ("நூற்றில்", "௧௦௦ இல்"),
                ("பத்தொன்பதில்", "௧௯ இல்"),
            ]
        )

        suffix_graph = pynutil.insert('integer: "') + suffix_graph + pynutil.insert('"')

        number_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')

        final_graph = number_graph | suffix_graph

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
