# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_HI_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. ऋण तेईस -> cardinal { integer: "२३" negative: "-" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        self.graph_two_digit = graph_teens_and_ties | (pynutil.insert("०") + graph_digit)
        graph_hundred = pynini.cross("सौ", "")
        delete_thousand = pynutil.delete("हज़ार") | pynutil.delete("हजार")
        graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("०"))
        graph_hundred_component += delete_space
        graph_hundred_component += self.graph_two_digit | pynutil.insert("००")

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_HI_DIGIT) + (NEMO_HI_DIGIT - "०") + pynini.closure(NEMO_HI_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundred_as_thousand = pynini.union(
            graph_teens_and_ties + delete_space + graph_hundred, pynutil.insert("०")
        )
        graph_hundred_as_thousand += delete_space
        graph_hundred_as_thousand += self.graph_two_digit | pynutil.insert("००")

        self.graph_hundreds = graph_hundred_component | graph_hundred_as_thousand

        graph_teens_and_ties_component = pynini.union(
            graph_teens_and_ties | pynutil.insert("00") + delete_space + (graph_digit | pynutil.insert("0")),
        )
        graph_ties_component_at_least_one_none_zero_digit = self.graph_two_digit @ (
            pynini.closure(NEMO_HI_DIGIT) + pynini.closure(NEMO_HI_DIGIT)
        )
        self.graph_ties_component_at_least_one_none_zero_digit = graph_ties_component_at_least_one_none_zero_digit

        # %% Indian numeric format simple https://en.wikipedia.org/wiki/Indian_numbering_system
        # This only covers "standard format".
        # Conventional format like thousand crores/lakh crores is yet to be implemented
        graph_in_thousands = pynini.union(
            self.graph_two_digit + delete_space + delete_thousand, pynutil.insert("००", weight=0.1),
        )
        self.graph_thousands = graph_in_thousands

        graph_in_lakhs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("लाख"), pynutil.insert("००", weight=0.1),
        )

        graph_in_crores = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("करोड़") | pynutil.delete("crores"),
            pynutil.insert("००", weight=0.1),
        )

        graph_in_arabs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("अरब"), pynutil.insert("००", weight=0.1),
        )

        graph_in_kharabs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("खरब"), pynutil.insert("००", weight=0.1),
        )

        graph_in_nils = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("नील"), pynutil.insert("००", weight=0.1),
        )

        graph_in_padmas = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("पद्म"), pynutil.insert("००", weight=0.1),
        )

        graph_in_shankhs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("शंख"), pynutil.insert("००", weight=0.1),
        )

        graph_ind = (
            graph_in_shankhs
            + delete_space
            + graph_in_padmas
            + delete_space
            + graph_in_nils
            + delete_space
            + graph_in_kharabs
            + delete_space
            + graph_in_arabs
            + delete_space
            + graph_in_crores
            + delete_space
            + graph_in_lakhs
            + delete_space
            + graph_in_thousands
        )
        graph_no_prefix = pynutil.add_weight(
            pynini.cross("सौ", "१००")
            | pynini.cross("हज़ार", "१०००")
            | pynini.cross("लाख", "१०००००")
            | pynini.cross("करोड़", "१०००००००"),
            2,
        )

        graph = pynini.union(
            graph_ind + delete_space + self.graph_hundreds, graph_zero, graph_no_prefix
        )  # graph_digit_plus_hundred,

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("०"))
            + pynini.difference(NEMO_HI_DIGIT, "०")
            + pynini.closure(NEMO_HI_DIGIT),
            "०",
        )

        labels_exception = [pynini.string_file(get_abs_path("data/numbers/labels_exception.tsv"))]

        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
