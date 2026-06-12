# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     [apache.org](http://www.apache.org/licenses/LICENSE-2.0)
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.kn.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_KN_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.kn.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals in Kannada
        e.g. ಋಣ ಇಪ್ಪತ್ಮೂರು -> cardinal { integer: "೨೩" negative: "-" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case

        # Load Kannada number mappings from TSV files
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()

        self.graph_zero = graph_zero
        self.graph_digit = graph_digit
        self.graph_single_digit_with_zero = pynutil.insert("೦") + graph_digit
        self.graph_teens_and_ties = graph_teens_and_ties
        self.graph_two_digit = graph_teens_and_ties | (pynutil.insert("೦") + graph_digit)

        # Kannada unit words
        graph_hundred = pynini.cross("ನೂರು", "")
        delete_hundred = pynutil.delete("ನೂರು") | pynutil.delete("ನೂರ")
        delete_thousand = pynutil.delete("ಸಾವಿರ") | pynutil.delete("ಸಾವಿರದ")

        # Hundreds component
        graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("೦"))
        graph_hundred_component += delete_space
        graph_hundred_component += self.graph_two_digit | pynutil.insert("೦೦")

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_KN_DIGIT) + (NEMO_KN_DIGIT - "೦") + pynini.closure(NEMO_KN_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        # Transducer for patterns like "ಹನ್ನೊಂದು ನೂರು" -> 1100
        graph_hundred_as_thousand = pynini.union(
            graph_teens_and_ties + delete_space + graph_hundred, pynutil.insert("೦")
        )
        graph_hundred_as_thousand += delete_space
        graph_hundred_as_thousand += self.graph_two_digit | pynutil.insert("೦೦")

        # Kannada fractional thousand patterns
        # ಒಂದೂವರೆ ಸಾವಿರ (one and a half thousand) -> 1500
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("ಒಂದೂವರೆ")
            + delete_space
            + pynutil.insert("೧೫೦೦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # ಎರಡೂವರೆ ಸಾವಿರ (two and a half thousand) -> 2500
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("ಎರಡೂವರೆ")
            + delete_space
            + pynutil.insert("೨೫೦೦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )

        # Fractional hundreds
        # ಒಂದೂವರೆ ನೂರು -> 150
        graph_in_hundreds = pynutil.add_weight(
            pynutil.delete("ಒಂದೂವರೆ")
            + delete_space
            + pynutil.insert("೧೫೦", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        # ಎರಡೂವರೆ ನೂರು -> 250
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("ಎರಡೂವರೆ")
            + delete_space
            + pynutil.insert("೨೫೦", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )

        self.graph_hundreds = graph_hundred_component | graph_hundred_as_thousand | graph_in_hundreds

        graph_ties_component_at_least_one_none_zero_digit = self.graph_two_digit @ (
            pynini.closure(NEMO_KN_DIGIT) + pynini.closure(NEMO_KN_DIGIT)
        )
        self.graph_ties_component_at_least_one_none_zero_digit = graph_ties_component_at_least_one_none_zero_digit

        # Indian numeric format for Kannada
        # Structure: Crores, Lakhs, Thousands, Hundreds (max unit is crore)
        graph_in_thousands = pynini.union(
            self.graph_two_digit + delete_space + delete_thousand,
            pynutil.insert("೦೦", weight=0.1),
        )
        self.graph_thousands = graph_in_thousands

        graph_in_lakhs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ಲಕ್ಷ"),
            pynutil.insert("೦೦", weight=0.1),
        )

        graph_in_crores = pynini.union(
            self.graph_two_digit + delete_space + (pynutil.delete("ಕೋಟಿ") | pynutil.delete("crores")),
            pynutil.insert("೦೦", weight=0.1),
        )

        # Build the full Indian number graph (crore is the highest unit)
        graph_ind = graph_in_crores + delete_space + graph_in_lakhs + delete_space + graph_in_thousands

        # Standalone unit words
        graph_no_prefix = pynutil.add_weight(
            pynini.cross("ನೂರು", "೧೦೦")
            | pynini.cross("ಸಾವಿರ", "೧೦೦೦")
            | pynini.cross("ಲಕ್ಷ", "೧೦೦೦೦೦")
            | pynini.cross("ಕೋಟಿ", "೧೦೦೦೦೦೦೦"),
            2,
        )

        graph = pynini.union(graph_ind + delete_space + self.graph_hundreds, graph_zero, graph_no_prefix)

        # Remove leading zeros
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("೦"))
            + pynini.difference(NEMO_KN_DIGIT, "೦")
            + pynini.closure(NEMO_KN_DIGIT),
            "೦",
        )

        # labels_exception = [pynini.string_file(get_abs_path("data/numbers/labels_exception.tsv"))]
        # graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = graph
        self.graph = graph  # (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        # Handle negative numbers (ಋಣ = negative in Kannada)
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
