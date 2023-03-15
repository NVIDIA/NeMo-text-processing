# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.sv.graph_utils import SV_UPPER
from pynini.lib import pynutil


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. "ABC" -> tokens { abbreviation { value: "A B C" } }

    Args:
        whitelist: whitelist FST
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, whitelist: 'pynini.FstLike', deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")
        # A.B.C. -> A. B. C.
        graph = SV_UPPER + dot + pynini.closure(insert_space + SV_UPPER + dot, 1)
        # A.B.C. -> A.B.C.
        graph |= SV_UPPER + dot + pynini.closure(SV_UPPER + dot, 1)
        # ABC -> A B C
        graph |= SV_UPPER + pynini.closure(insert_space + SV_UPPER, 1)

        # exclude words that are included in the whitelist
        if whitelist is not None:
            graph = pynini.compose(
                pynini.difference(pynini.project(graph, "input"), pynini.project(whitelist.graph, "input")), graph
            )

        graph = pynutil.insert("value: \"") + graph.optimize() + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
