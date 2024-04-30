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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal e.g.
        tokens { ordinal { integer: "第一千万" } } -> 第一千万
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)

        symbol = pynini.union("-", "~", "——", "—")
        dash = pynini.cross(symbol, "到")

        delete_morpheme = (
            pynutil.delete("range: \"") + (pynini.closure('从') | (pynini.closure('到') | dash)) + pynutil.delete("\"")
        )
        graph_integer = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure("第", 0, 1)
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        graph_range = (
            pynini.closure(delete_morpheme, 0, 1)
            + pynini.closure(delete_space, 0, 1)
            + graph_integer
            + delete_space
            + delete_morpheme
            + delete_space
            + graph_integer
        )

        final_graph = pynutil.add_weight(graph_integer, -2.0) | graph_range

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
