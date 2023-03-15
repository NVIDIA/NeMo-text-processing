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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. hundraandra -> tokens { name: "102." }

    Args:
        itn_cardinal_tagger: ITN Cardinal Tagger
        tn_ordinal_verbalizer: TN Ordinal Verbalizer
    """

    def __init__(self, tn_ordinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        graph = pynini.arcmap(tn_ordinal.bare_ordinals, map_type="rmweight").invert().optimize()

        final_graph = graph + pynutil.insert(".")
        self.graph = final_graph

        graph = pynutil.insert("name: \"") + final_graph + pynutil.insert("\"")
        self.fst = graph.optimize()
