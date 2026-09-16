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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.ta.graph_utils import RANGE_WORD, rank
from nemo_text_processing.text_normalization.ta.taggers.decimal import quantity_words


class RangeFst(GraphFst):
    """
    Finite state transducer for classifying numeric ranges, e.g.
        10-20 -> tokens { name: "பத்து முதல் இருபது" }
        10-20ல் -> tokens { name: "பத்து முதல் இருபதில்" }
        5-10 லட்சம் -> tokens { name: "ஐந்து முதல் பத்து லட்சம்" }

    A glued case suffix on the upper bound and a scale word after it belong to the range.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        spaced, _, _ = quantity_words()
        # A scale word after the upper bound belongs to the range, not to a decimal quantity.
        quantity = pynini.accep(" ") + spaced
        graph = (
            cardinal.final_graph
            + pynutil.delete(pynini.closure(" ", 0, 1) + "-" + pynini.closure(" ", 0, 1))
            + pynutil.insert(f" {RANGE_WORD} ")
            + (cardinal.final_graph | cardinal.suffixed_graph + rank(0.1))
            + pynini.closure(quantity, 0, 1)
        )
        self.graph = convert_space(graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
