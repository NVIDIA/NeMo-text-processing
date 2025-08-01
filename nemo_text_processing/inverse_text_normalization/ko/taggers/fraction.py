# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_SPACE, GraphFst


class FractionFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        """
        Fitite state transducer for classifying fractions
        e.g.,
        fraction { denominator: "사" numerator: "삼" } -> 3/4
        fraction { mixed number: "일" denominator: "사" numerator: "삼" } -> 1 3/4
        fraction { denominator: "루트삼" numerator: "일" } -> 1/√3
        fraction { denominator: "일점육오" numerator: "오십" } -> 50/1.65
        fraction { denominator: "이루트육" numerator: "삼" } -> 3/2√6
        """
        super().__init__(name="fraction", kind="classify")

        cardinal = cardinal.just_cardinals
        decimal = decimal.just_decimal

        # Expression between fraction. Means the dash "/"
        fraction_word = pynutil.delete("분의")
        # Expression combining mixed number and fraction. Optional to use
        connecting_word = pynutil.delete("와") | pynutil.delete("과")
        # Expression for "√"
        root_word = pynini.accep("√") | pynini.cross("루트", "√")

        graph_sign = (
            pynutil.insert("negative: \"") + (pynini.accep("-") | pynini.cross("마이너스", "-")) + pynutil.insert("\"")
        )

        # graph_mixed_number considers all of possible combination number you can have in front of fraction
        graph_mixed_number = (
            pynutil.insert("integer_part: \"")
            + (
                decimal
                | (decimal + connecting_word)
                | (root_word + decimal)
                | (cardinal + root_word + decimal)
                | (root_word + decimal + connecting_word)
                | (cardinal + root_word + decimal + connecting_word)
                | cardinal
                | (cardinal + connecting_word)
                | (root_word + cardinal)
                | (cardinal + root_word + cardinal)
                | (root_word + cardinal + connecting_word)
                | (cardinal + root_word + cardinal + connecting_word)
            )
            + pynutil.insert("\"")
        )

        graph_denominator = (
            pynutil.insert("denominator: \"")
            + (
                (
                    decimal
                    | (cardinal + root_word + decimal)
                    | (root_word + decimal)
                    | cardinal
                    | (cardinal + root_word + cardinal)
                    | (root_word + cardinal)
                )
                + pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1)
            )
            + pynutil.insert("\"")
        )

        graph_numerator = (
            pynutil.insert("numerator: \"")
            + (
                (
                    decimal
                    | (cardinal + root_word + decimal)
                    | (root_word + decimal)
                    | cardinal
                    | (cardinal + root_word + cardinal)
                    | (root_word + cardinal)
                )
                + pynini.closure(pynutil.delete(NEMO_SPACE))
            )
            + pynutil.insert("\"")
        )

        graph_fraction_sign = (
            graph_sign
            + pynutil.insert(NEMO_SPACE)
            + graph_denominator
            + pynutil.insert(NEMO_SPACE)
            + fraction_word
            + graph_numerator
        )
        graph_fraction_no_sign = graph_denominator + pynutil.insert(NEMO_SPACE) + fraction_word + graph_numerator
        # Only fraction like "1/3" or "- 1/3"
        graph_fractions = graph_fraction_sign | graph_fraction_no_sign
        # Mixed number fraction like "2 1/3" or "-2 1/3"
        graph_mixed_number_fraction = (
            pynini.closure((graph_sign + pynutil.insert(" ")), 0, 1)
            + pynutil.add_weight(graph_mixed_number, 1.1)
            + pynutil.insert(NEMO_SPACE)
            + graph_denominator
            + pynutil.insert(NEMO_SPACE)
            + fraction_word
            + graph_numerator
        )

        final_graph = graph_fractions | graph_mixed_number_fraction

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
