# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.mr.graph_utils import (
    MINUS,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.mr.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. तेहतीस -> cardinal { integer: "३३" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digits = pynini.string_file(get_abs_path("data/numbers/digits.tsv")).invert()
        graph_tens = pynini.string_file(get_abs_path("data/numbers/tens.tsv")).invert()
        graph_hundred_unique = pynini.string_file(get_abs_path("data/numbers/hundred.tsv")).invert()

        graph_hundred = pynini.cross("शे", "")

        graph_hundred_component = pynini.union(graph_digits + graph_hundred, pynutil.insert("०"))
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(pynutil.insert("००"), graph_tens, pynutil.insert("०") + graph_digits)

        graph_hundred_component_at_least_one_non_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "०") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_non_zero_digit = graph_hundred_component_at_least_one_non_zero_digit

        # eleven hundred -> 1100 etc form
        graph_hundred_as_thousand = graph_tens + graph_hundred
        graph_hundred_as_thousand += delete_space + pynini.union(
            pynutil.insert("००"), graph_tens, pynutil.insert("०") + graph_digits
        )

        graph_hundreds = graph_hundred_component | graph_hundred_as_thousand

        graph_two_digit_component = pynini.union(pynutil.insert("००"), graph_tens, pynutil.insert("०") + graph_digits)

        graph_two_digit_component_at_least_one_non_zero_digit = graph_two_digit_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "०") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_two_digit_component_at_least_one_non_zero_digit = (
            graph_two_digit_component_at_least_one_non_zero_digit
        )

        graph_thousands = pynini.union(
            graph_two_digit_component_at_least_one_non_zero_digit + delete_space + pynutil.delete("हजार"),
            pynutil.insert("००", weight=0.1),
        )

        graph_lakhs = pynini.union(
            graph_two_digit_component_at_least_one_non_zero_digit + delete_space + pynutil.delete("लाख"),
            pynutil.insert("००", weight=0.1),
        )

        graph_crores = pynini.union(
            graph_two_digit_component_at_least_one_non_zero_digit + delete_space + pynutil.delete("कोटी"),
            pynutil.insert("००", weight=0.1),
        )

        graph_arabs = pynini.union(
            graph_two_digit_component_at_least_one_non_zero_digit + delete_space + pynutil.delete("अब्ज"),
            pynutil.insert("००", weight=0.1),
        )

        graph_higher_powers = (
            graph_arabs + delete_space + graph_crores + delete_space + graph_lakhs + delete_space + graph_thousands
        )

        graph = pynini.union(graph_higher_powers + delete_space + graph_hundreds, graph_hundred_unique, graph_zero,)

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("०")) + pynini.difference(NEMO_DIGIT, "०") + pynini.closure(NEMO_DIGIT), "०"
        )
        graph = graph.optimize()

        self.graph = (pynini.project(graph, "input")) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
