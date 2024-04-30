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


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal, e.g.
        cardinal { negative: "负" integer: "23" } -> 负二十三
        cardinal { integer: "23" } -> 二十三
        cardinal { positive: "正" integer: "23" } -> 正二十三
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        delete_sign = pynini.cross("negative: \"-\"", "负")
        delete_integer = (
            pynutil.delete("integer: ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
        )
        delete_mandarin = pynutil.delete("quantity: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        graph_mandarin = (delete_integer + delete_mandarin) | (
            delete_sign + delete_space + delete_integer + delete_mandarin
        )
        graph_sign = delete_sign + delete_space + delete_integer
        final_graph = delete_integer | graph_sign | graph_mandarin
        self.numbers = final_graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
