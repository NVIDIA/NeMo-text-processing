# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Korean ordinal expressions, e.g.
    1번째 -> ordinal { integer: "첫번째" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        graph_ordinal_1to39 = pynini.string_file(get_abs_path("data/ordinal/digit_1to39.tsv")) + pynini.accep("번째")

        graph_cardinal = cardinal.just_cardinals + pynini.accep("번째")

        graph_ordinal = (
            pynutil.add_weight(graph_ordinal_1to39, 0.1) | pynutil.add_weight(graph_cardinal, 1.0)
        ).optimize()

        final_graph = pynutil.insert('integer: "') + graph_ordinal + pynutil.insert('"')

        graph_ordinal_final = self.add_tokens(final_graph)
        self.fst = graph_ordinal_final.optimize()
