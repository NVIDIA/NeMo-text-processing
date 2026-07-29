# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_CHAR,
    NEMO_NOT_SPACE,
    NEMO_SPACE,
    NEMO_TE_DIGIT,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu cardinals, e.g.
        ఇరవై మూడు -> cardinal { integer: "౨౩" }
        మైనస్ ఇరవై మూడు -> cardinal { integer: "౨౩" negative: "-" }


    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        graph_ties_prefix = pynini.string_file(get_abs_path("data/numbers/ties_prefix.tsv")).invert()

        graph_people = pynini.string_file(get_abs_path("data/numbers/digit_people.tsv")).invert()
        graph_special = pynini.string_file(get_abs_path("data/numbers/special_numbers.tsv")).invert()
        two_te_digits = NEMO_TE_DIGIT + NEMO_TE_DIGIT
        graph_digit |= (graph_people | graph_special) @ NEMO_TE_DIGIT

        self.graph_zero = graph_zero
        self.graph_digit = graph_digit

        graph_two_digit = (
            graph_teens_and_ties
            | (graph_ties_prefix + delete_space + graph_digit)
            | (pynutil.insert("౦") + graph_digit)
            | ((graph_people | graph_special) @ two_te_digits)
        )
        self.graph_two_digit = graph_two_digit
        two_digit_or_zeros = graph_two_digit | pynutil.insert("౦౦")

        graph_case_suffix = pynini.string_file(get_abs_path("data/numbers/case_suffix.tsv"))
        optional_case = pynini.closure(graph_case_suffix, 0, 1)

        optional_one = pynini.closure(pynutil.delete("ఒక") + delete_space, 0, 1)

        graph_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv")).invert()
        delete_hundred = graph_hundred @ pynini.accep("")
        one_hundred_value = graph_hundred @ pynini.closure(NEMO_TE_DIGIT, 1)

        graph_thousand = pynini.string_file(get_abs_path("data/numbers/thousand.tsv")).invert()
        delete_thousand = graph_thousand @ pynini.accep("")
        one_thousand_value = graph_thousand @ pynini.closure(NEMO_TE_DIGIT, 1)

        graph_lakh = pynini.string_file(get_abs_path("data/numbers/lakh.tsv")).invert()
        delete_lakh = graph_lakh @ pynini.accep("")
        one_lakh_value = graph_lakh @ pynini.closure(NEMO_TE_DIGIT, 1)

        graph_crore = pynini.string_file(get_abs_path("data/numbers/crore.tsv")).invert()
        delete_crore = graph_crore @ pynini.accep("")
        one_crore_value = graph_crore @ pynini.closure(NEMO_TE_DIGIT, 1)

        hundred_prefix = (
            (graph_digit + delete_space + delete_hundred) | (optional_one + one_hundred_value) | pynutil.insert("౦")
        )
        graph_hundreds = hundred_prefix + delete_space + two_digit_or_zeros
        self.graph_hundreds = graph_hundreds

        thousand_block = (
            (graph_two_digit + delete_space + delete_thousand)
            | (optional_one + one_thousand_value)
            | pynutil.insert("౦౦")
        )

        lakh_block = (
            (graph_two_digit + delete_space + delete_lakh) | (optional_one + one_lakh_value) | pynutil.insert("౦౦")
        )

        graph_below_crore = lakh_block + delete_space + thousand_block + delete_space + graph_hundreds

        at_least_one_non_zero = pynini.closure(NEMO_TE_DIGIT) + (NEMO_TE_DIGIT - "౦") + pynini.closure(NEMO_TE_DIGIT)

        crore_block = (
            ((graph_below_crore @ at_least_one_non_zero) + delete_space + delete_crore)
            | (optional_one + one_crore_value)
            | pynutil.insert("౦౦౦౦౦౦౦")
        )

        graph_full = crore_block + delete_space + graph_below_crore
        strip_leading_zeros = (
            pynutil.delete(pynini.closure("౦")) + (NEMO_TE_DIGIT - "౦") + pynini.closure(NEMO_TE_DIGIT)
        )

        no_trailing_space = pynini.closure(NEMO_CHAR) + NEMO_NOT_SPACE
        graph_number = (no_trailing_space @ graph_full) @ strip_leading_zeros

        graph_leading_zeros = graph_zero + pynini.closure(delete_space + (graph_zero | graph_digit), 1)

        graph = (graph_number | graph_zero) + optional_case
        graph |= graph_leading_zeros

        self.graph_no_exception = graph

        self.graph = graph.optimize()

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE,
            0,
            1,
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
