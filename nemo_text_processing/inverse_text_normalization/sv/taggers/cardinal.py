# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below ten are not converted. 
    Allows both compound numeral strings or separated by whitespace.

        e.g. minus tjugoen -> cardinal { negative: "-" integer: "21" } }
        e.g. minus tjugoett -> cardinal { negative: "-" integer: "21" } }

    Args:
        tn_cardinal_tagger: TN cardinal tagger
    """

    def __init__(self, tn_cardinal_tagger: GraphFst):
        super().__init__(name="cardinal", kind="classify")

        graph = pynini.invert(pynini.arcmap(tn_cardinal_tagger.graph, map_type="rmweight")).optimize()

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            (tn_cardinal_tagger.graph_hundreds_component_at_least_one_non_zero_digit).invert().optimize()
        )

        self.graph_ties = (tn_cardinal_tagger.two_digit_non_zero).invert().optimize()
        # this is to make sure if there is an ambiguity with decimal, decimal is chosen, e.g. 1000000 vs. 1 million
        # graph = pynutil.add_weight(graph, weight=0.001)
        self.graph_no_exception = graph
        self.digit = pynini.arcmap(tn_cardinal_tagger.digit, map_type="rmweight").invert().optimize()
        graph_exception = pynini.project(self.digit, 'input')
        # self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        self.optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus ", "\"-\" "), 0, 1
        )

        final_graph = self.optional_minus_graph + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()