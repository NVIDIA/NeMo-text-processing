# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "२३ ४/६" ->
    fraction { integer: "तेईस" numerator: "चार" denominator: "छः"}
    ४/६" ->
    fraction { numerator: "चार" denominator: "छः"}


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
        )
        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.numerator = (
            pynutil.insert("numerator: \"") + cardinal_graph + pynini.cross(pynini.union("/", " / "), "\" ")
        )
        self.denominator = pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")

        self.graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(" "), 0, 1)
            + self.numerator
            + self.denominator
        )

        graph = self.graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
