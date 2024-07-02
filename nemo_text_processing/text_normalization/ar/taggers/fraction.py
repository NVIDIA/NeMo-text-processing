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

from nemo_text_processing.text_normalization.ar.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ar.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "1 1/2" ->
    tokens { fraction { integer_part: "واحد" numerator: "واحد" denominator: "نص" } }

    Args:
        cardinal: cardinal fst
    """

    def __init__(self, cardinal):
        super().__init__(name="fraction", kind="classify")

        cardinal_graph = cardinal.cardinal_numbers
        digit_one = pynini.accep("1")
        digit_two = pynini.accep("2")
        digit_three_to_ten = pynini.union("3", "4", "5", "6", "7", "8", "9", "10")
        digit_one_to_ten = pynini.union("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")

        graph_ones = pynini.string_file(get_abs_path("data/number/fraction_singular.tsv")).optimize()
        graph_dual = pynini.string_file(get_abs_path("data/number/fraction_dual.tsv")).optimize()
        graph_plural = pynini.string_file(get_abs_path("data/number/fraction_plural.tsv")).optimize()

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        dividor = pynini.cross("/", "\" ") | pynini.cross(" / ", "\" ")
        graph_numerator = pynutil.insert("numerator: \"") + cardinal_graph

        denominator_singual = pynutil.insert("denominator: \"") + digit_one_to_ten @ graph_ones + pynutil.insert("\"")
        denominator_dual = pynutil.insert("denominator: \"") + digit_one_to_ten @ graph_dual + pynutil.insert("\"")
        denominator_plural = pynutil.insert("denominator: \"") + digit_one_to_ten @ graph_plural + pynutil.insert("\"")

        numerator_one = digit_one @ graph_numerator + dividor + denominator_singual
        numerator_two = digit_two @ graph_numerator + dividor + denominator_dual
        numerator_three_to_ten = digit_three_to_ten @ graph_numerator + dividor + denominator_plural

        numerator_more_than_ten = (
            graph_numerator + dividor + pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")
        )
        fraction_graph = (
            numerator_one | numerator_two | numerator_three_to_ten | pynutil.add_weight(numerator_more_than_ten, 0.001)
        )
        graph = pynini.closure(integer + pynini.accep(" "), 0, 1) + (fraction_graph)
        graph |= pynini.closure(integer + (pynini.accep(" ") | pynutil.insert(" ")), 0, 1) + pynini.compose(
            pynini.string_file(get_abs_path("data/number/fraction.tsv")), (fraction_graph)
        )
        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
