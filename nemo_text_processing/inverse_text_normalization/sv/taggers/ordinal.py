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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. hundraandra -> tokens { name: "102." }

    Args:
        tn_ordinal: TN Ordinal Tagger
    """

    def __init__(self, tn_ordinal: GraphFst, project_input: bool = False):
        super().__init__(name="ordinal", kind="classify", project_input=project_input)

        # Invert the TN ordinal graph to map from ordinal words to numbers
        inverted_graph = pynini.arcmap(tn_ordinal.bare_ordinals, map_type="rmweight").invert().optimize()
        self.bare_ordinals = inverted_graph
        self.ordinals = inverted_graph + pynutil.insert(".")

        # Exclude "första" and "andra" from the ordinal tagger - they should be handled by whitelist/word tagger
        forsta_andra = pynini.project(pynini.union("1", "2") @ tn_ordinal.bare_ordinals, "output")
        
        # Only accept ordinals other than "första"/"andra"
        ordinal_graph = (pynini.project(inverted_graph, "input") - forsta_andra.arcsort()) @ inverted_graph + pynutil.insert(".")

        # Wrap in integer field
        graph = pynutil.insert("integer: \"") + ordinal_graph + pynutil.insert("\"")
        
        # Add ordinal tokens wrapper
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
