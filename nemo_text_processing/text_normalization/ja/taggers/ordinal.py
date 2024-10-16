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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal, e.g.
    第100 -> ordinal { integer: "第百" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.just_cardinals
        morpheme_pre = pynini.accep('第')
        morpheme_post = pynini.accep('番目')
        graph_ordinal = pynini.union(morpheme_pre + graph_cardinal, graph_cardinal + morpheme_post)

        final_graph = pynutil.insert("integer: \"") + graph_ordinal + pynutil.insert("\"")

        graph_ordinal_final = self.add_tokens(final_graph)
        self.fst = graph_ordinal_final.optimize()
