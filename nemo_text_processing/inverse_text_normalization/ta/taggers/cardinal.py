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
# limitations under the License.

import pynini
from pynini.lib import pynutil

import nemo_text_processing.inverse_text_normalization.ta.utils
from nemo_text_processing.inverse_text_normalization.ta.graph_utils import NEMO_SIGMA, GraphFst, delete_space


class CardinalFst(GraphFst):

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        graph_zero_raw = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/zero.tsv")
        )
        graph_digit_raw = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/digit.tsv")
        )
        graph_teens_and_ties_raw = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/teens_and_ties.tsv")
        )
        graph_hundreds_raw = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/hundreds.tsv")
        )
        graph_thousands_raw = pynini.string_file(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/thousands.tsv")
        )

        hundred_join_pairs = []

        with open(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/hundreds.tsv"),
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                numeral, word = line.split("\t")

                if word.endswith("று"):
                    word = word.removesuffix("று") + "ற்று"
                elif word.endswith("ம்"):
                    word = word.removesuffix("ம்") + "த்து"

                hundred_join_pairs.append((word, numeral))

        graph_hundreds_join = pynini.string_map(hundred_join_pairs)

        thousand_join_pairs = []

        with open(
            nemo_text_processing.inverse_text_normalization.ta.utils.get_abs_path("data/numbers/thousands.tsv"),
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                value, word = line.split("\t")

                if word.endswith("ம்"):
                    join_word = word[:-1] + "த்து"
                    thousand_join_pairs.append((join_word, value))

        graph_all_thousand_join = pynini.string_map(thousand_join_pairs)

        graph_zero = graph_zero_raw.copy().invert()
        graph_digit = graph_digit_raw.copy().invert()
        graph_teens_and_ties = graph_teens_and_ties_raw.copy().invert()
        graph_hundreds = graph_hundreds_raw.copy().invert()
        graph_thousands = graph_thousands_raw.copy().invert()

        self.graph_single_digit = graph_digit

        graph_numeric_digit = pynini.union(*list("௦௧௨௩௪௫௬௭௮௯"))
        graph_numeric_two_digit = graph_numeric_digit + graph_numeric_digit

        graph_join_digit = pynini.union(
            "௧",
            "௨",
            "௩",
            "௪",
            "௫",
            "௬",
            "௭",
            "௮",
            "௯",
        )

        graph_ties_join = graph_teens_and_ties @ graph_join_digit
        graph_two_digit_composed = graph_ties_join + delete_space + graph_digit

        fusion_rules = pynini.union(
            pynini.cross("த்தை", "த்து ஐ"),
            pynini.cross("த்தா", "த்து ஆ"),
            pynini.cross("த்தே", "த்து ஏ"),
            pynini.cross("த்தெ", "த்து எ"),
            pynini.cross("த்தொ", "த்து ஒ"),
            pynini.cross("ற்றா", "ற்று ஆ"),
        )

        fusion_rewrite = pynini.cdrewrite(fusion_rules, "", "", NEMO_SIGMA)
        graph_two_digit_fused = fusion_rewrite @ graph_two_digit_composed

        self.graph_two_digit = (graph_teens_and_ties | graph_two_digit_composed | graph_two_digit_fused).optimize()

        self.graph_exact_hundreds = graph_hundreds + pynutil.insert("௦௦")
        self.graph_hundred_join = graph_hundreds_join

        graph_digit_with_zero = pynutil.insert("௦") + graph_digit

        graph_hundred_with_digit = self.graph_hundred_join + delete_space + graph_digit_with_zero

        graph_hundred_with_two_digit = self.graph_hundred_join + delete_space + self.graph_two_digit

        graph_hundred_with_numeric_two_digit = self.graph_hundred_join + delete_space + graph_numeric_two_digit

        self.graph_hundred_with_remainder = (
            graph_hundred_with_digit | graph_hundred_with_two_digit | graph_hundred_with_numeric_two_digit
        ).optimize()

        graph_single_thousand_hundred_remainder = (
            graph_all_thousand_join + delete_space + self.graph_hundred_with_remainder
        ).optimize()

        self.graph_exact_thousands = graph_thousands + pynutil.insert("௦௦௦")

        graph_thousand_with_digit = graph_all_thousand_join + pynutil.insert("௦௦") + delete_space + graph_digit

        graph_thousand_with_two_digit = (
            graph_all_thousand_join + pynutil.insert("௦") + delete_space + self.graph_two_digit
        )

        graph_thousand_with_hundred = graph_all_thousand_join + delete_space + self.graph_exact_hundreds

        graph_thousand_with_hundred_remainder = (
            graph_all_thousand_join + delete_space + self.graph_hundred_with_remainder
        )

        self.graph_thousand_with_remainder = (
            graph_thousand_with_digit
            | graph_thousand_with_two_digit
            | graph_thousand_with_hundred
            | graph_thousand_with_hundred_remainder
        ).optimize()

        graph_lakh_word = pynini.union(
            pynini.cross("இலட்சம்", ""),
            pynini.cross("லட்சம்", ""),
        )

        graph_single_lakh = graph_digit + delete_space + graph_lakh_word + pynutil.insert("௦௦௦௦௦")

        graph_two_digit_lakh = self.graph_two_digit + delete_space + graph_lakh_word + pynutil.insert("௦௦௦௦௦")

        self.graph_lakh = (graph_single_lakh | graph_two_digit_lakh).optimize()

        graph_lakh_join_delete = pynini.union(
            pynini.cross("இலட்சத்து", ""),
            pynini.cross("லட்சத்து", ""),
        )

        graph_lakh_multiplier = graph_digit | self.graph_two_digit

        graph_lakh_with_digit = (
            graph_lakh_multiplier
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦௦௦")
            + delete_space
            + graph_digit
        )

        graph_lakh_with_two_digit = (
            graph_lakh_multiplier
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦௦")
            + delete_space
            + self.graph_two_digit
        )

        graph_lakh_with_hundred = (
            graph_lakh_multiplier
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_exact_hundreds
        )

        graph_lakh_with_hundred_remainder = (
            graph_lakh_multiplier
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_hundred_with_remainder
        )

        graph_lakh_single_with_thousand = (
            graph_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦")
            + delete_space
            + self.graph_thousand_with_remainder
        )

        graph_lakh_two_digit_with_thousand = (
            self.graph_two_digit
            + delete_space
            + graph_lakh_join_delete
            + delete_space
            + self.graph_thousand_with_remainder
        )

        graph_lakh_with_thousand = graph_lakh_single_with_thousand | graph_lakh_two_digit_with_thousand

        self.graph_lakh_with_remainder = (
            graph_lakh_with_digit
            | graph_lakh_with_two_digit
            | graph_lakh_with_hundred
            | graph_lakh_with_hundred_remainder
            | graph_lakh_with_thousand
        ).optimize()

        graph_crore_word = pynini.union(pynini.cross("கோடி", ""))

        graph_single_crore = graph_digit + delete_space + graph_crore_word + pynutil.insert("௦௦௦௦௦௦௦")

        graph_two_digit_crore = self.graph_two_digit + delete_space + graph_crore_word + pynutil.insert("௦௦௦௦௦௦௦")

        self.graph_crore = (graph_single_crore | graph_two_digit_crore).optimize()

        graph_crore_join_delete = pynini.union(pynini.cross("கோடியே", ""))

        graph_crore_single_thousand_hundred_remainder = (
            graph_single_thousand_hundred_remainder + delete_space + pynini.cross("கோடியே", "")
        ).optimize()

        graph_crore_multiplier = (
            graph_crore_single_thousand_hundred_remainder
            | graph_digit
            | self.graph_two_digit
            | self.graph_exact_hundreds
            | self.graph_hundred_with_remainder
            | self.graph_exact_thousands
            | self.graph_thousand_with_remainder
            | self.graph_lakh
            | self.graph_lakh_with_remainder
        ).optimize()

        graph_crore_with_digit = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦௦௦௦௦௦")
            + delete_space
            + graph_digit
        )

        graph_crore_with_two_digit = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦௦௦௦௦")
            + delete_space
            + self.graph_two_digit
        )

        graph_crore_with_hundred = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦௦௦௦")
            + delete_space
            + self.graph_exact_hundreds
        )

        graph_crore_with_hundred_remainder = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦௦௦௦")
            + delete_space
            + self.graph_hundred_with_remainder
        )

        graph_crore_with_thousand_remainder = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦௦௦")
            + delete_space
            + self.graph_thousand_with_remainder
        )

        graph_crore_with_exact_lakh = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_lakh
        )

        graph_single_lakh_with_digit = (
            graph_digit + delete_space + graph_lakh_join_delete + pynutil.insert("௦௦௦௦") + delete_space + graph_digit
        )

        graph_single_lakh_with_two_digit = (
            graph_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦௦")
            + delete_space
            + self.graph_two_digit
        )

        graph_single_lakh_with_hundred = (
            graph_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_exact_hundreds
        )

        graph_single_lakh_with_hundred_remainder = (
            graph_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_hundred_with_remainder
        )

        graph_single_lakh_with_thousand = graph_lakh_single_with_thousand

        graph_single_lakh_remainder = (
            graph_single_lakh_with_digit
            | graph_single_lakh_with_two_digit
            | graph_single_lakh_with_hundred
            | graph_single_lakh_with_hundred_remainder
            | graph_single_lakh_with_thousand
        ).optimize()

        graph_two_digit_lakh_with_digit = (
            self.graph_two_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦௦௦")
            + delete_space
            + graph_digit
        )

        graph_two_digit_lakh_with_two_digit = (
            self.graph_two_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦௦")
            + delete_space
            + self.graph_two_digit
        )

        graph_two_digit_lakh_with_hundred = (
            self.graph_two_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_exact_hundreds
        )

        graph_two_digit_lakh_with_hundred_remainder = (
            self.graph_two_digit
            + delete_space
            + graph_lakh_join_delete
            + pynutil.insert("௦௦")
            + delete_space
            + self.graph_hundred_with_remainder
        )

        graph_two_digit_lakh_with_thousand = (
            self.graph_two_digit
            + delete_space
            + graph_lakh_join_delete
            + delete_space
            + self.graph_thousand_with_remainder
        )

        graph_two_digit_lakh_remainder = (
            graph_two_digit_lakh_with_digit
            | graph_two_digit_lakh_with_two_digit
            | graph_two_digit_lakh_with_hundred
            | graph_two_digit_lakh_with_hundred_remainder
            | graph_two_digit_lakh_with_thousand
        ).optimize()

        graph_crore_with_single_lakh_remainder = (
            graph_crore_multiplier
            + delete_space
            + graph_crore_join_delete
            + pynutil.insert("௦")
            + delete_space
            + graph_single_lakh_remainder
        )

        self.graph_crore_with_remainder = (
            graph_crore_with_digit
            | graph_crore_with_two_digit
            | graph_crore_with_hundred
            | graph_crore_with_hundred_remainder
            | graph_crore_with_thousand_remainder
            | graph_crore_with_exact_lakh
            | graph_crore_with_single_lakh_remainder
            | (
                graph_crore_multiplier
                + delete_space
                + graph_crore_join_delete
                + delete_space
                + graph_two_digit_lakh_remainder
            )
        ).optimize()

        graph = (
            graph_zero
            | graph_digit
            | self.graph_two_digit
            | self.graph_exact_hundreds
            | self.graph_hundred_with_remainder
            | self.graph_exact_thousands
            | self.graph_thousand_with_remainder
            | self.graph_lakh
            | self.graph_lakh_with_remainder
            | self.graph_crore
            | self.graph_crore_with_remainder
        ).optimize()

        number_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        final_graph = self.add_tokens(number_graph)
        self.fst = final_graph.optimize()
