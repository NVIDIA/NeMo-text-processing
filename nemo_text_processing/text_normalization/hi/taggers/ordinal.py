# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_HI_DIGIT, GraphFst
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Hindi ordinals, e.g.
        १०वां -> ordinal { integer: "दसवां" }
        २१वीं -> ordinal { integer: "इक्कीसवीं" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst | None = None, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        cardinal = cardinal if cardinal is not None else CardinalFst(deterministic=deterministic)

        suffixes_fst = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))

        number = pynini.closure(NEMO_HI_DIGIT, 1)
        # Build graph by mapping numbers with suffixes through cardinal FST
        graph = (number + suffixes_fst) @ ((number @ cardinal.final_graph) + suffixes_fst)
        graph = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
