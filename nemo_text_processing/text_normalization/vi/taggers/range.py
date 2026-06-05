# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_SPACE, GraphFst


class RangeFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese ranges with dash "-"
    Examples:
        10k-20k -> tokens { name: "mười nghìn đến hai mười nghìn" }
        10h-8h -> tokens { name: "mười giờ đến tám giờ" }
        10$-20$ -> tokens { name: "mười đô la đến hai mười đô la" }

    Args:
        time: composed time tagger and verbalizer
        date: composed date tagger and verbalizer
        decimal: composed decimal tagger and verbalizer
        money: composed money tagger and verbalizer
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self,
        time: GraphFst,
        date: GraphFst,
        decimal: GraphFst,
        money: GraphFst,
        measure: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        delete_space = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1)

        # Pattern: X-Y -> X đến Y
        # This will handle time ranges, date ranges, decimal ranges, and money ranges with dash
        range_pattern = (
            (time | date | decimal | money | measure)
            + delete_space
            + pynini.cross("-", " đến ")
            + delete_space
            + (time | date | decimal | money | measure)
        )

        self.graph = range_pattern

        # Convert to final FST format
        self.graph = self.graph.optimize()
        graph = pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")
        self.fst = graph.optimize()
