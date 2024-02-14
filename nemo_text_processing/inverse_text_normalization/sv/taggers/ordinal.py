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
        tn_ordinal_verbalizer: TN Ordinal Verbalizer
    """

    def __init__(self, tn_ordinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        graph = pynini.arcmap(tn_ordinal.bare_ordinals, map_type="rmweight").invert().optimize()
        self.bare_ordinals = graph
        self.ordinals = graph + pynutil.insert(".")

        forsta_andra = pynini.project(pynini.union("1", "2") @ tn_ordinal.bare_ordinals, "output")
        graph = ((pynini.project(graph, "input") - forsta_andra.arcsort()) @ graph) + pynutil.insert(".")

        graph = pynutil.insert("name: \"") + graph + pynutil.insert("\"")
        self.fst = graph.optimize()
