# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_ALL_ZERO
from nemo_text_processing.text_normalization.ta.taggers.cardinal import ORDINAL_MARKERS


def first_ordinal() -> 'pynini.FstLike':
    """
    முதல் is the idiomatic stem for first (1வது -> முதலாவது, 1ஆம் -> முதலாம்).
    """
    return pynini.union(pynini.cross("௧", "முதலா"), pynini.cross("1", "முதலா")) + ORDINAL_MARKERS


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals, e.g.
        5வது -> ordinal { integer: "ஐந்தாவது" }
        1வது -> ordinal { integer: "முதலாவது" }
        28ம் -> ordinal { integer: "இருபத்தெட்டாம்" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        graph = cardinal.ordinal_graph(cardinal.final_graph)
        graph = pynini.union(graph, pynutil.add_weight(first_ordinal(), -0.1))
        # A leading zero on an ordinal is not spoken: 01ஆம் is முதலாம் and 007ஆம் ஏழாம், never
        # பூஜ்யம் ஒன்றாம். Stripping outranks the cardinal's leading-zero reading; 0வது stays.
        stripped = pynutil.delete(pynini.closure(NEMO_ALL_ZERO, 1)) + graph
        graph = pynini.union(graph, pynutil.add_weight(stripped, -1.0))

        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph).optimize()
