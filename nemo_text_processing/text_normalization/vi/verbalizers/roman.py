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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing Roman numerals in Vietnamese
        e.g. tokens { roman { key_cardinal: "thế kỉ" integer: "mười lăm" } } -> thế kỉ mười lăm
        e.g. tokens { roman { key_cardinal: "thế kỷ" integer: "bốn" } } -> thế kỷ bốn
        e.g. tokens { roman { key_cardinal: "thứ" integer: "bốn" } } -> thứ bốn
        e.g. tokens { roman { integer: "mười lăm" } } -> mười lăm

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)

        key_cardinal = pynutil.delete("key_cardinal: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        integer = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_with_key = key_cardinal + delete_space + pynutil.insert(NEMO_SPACE) + integer
        graph_without_key = integer
        graph = pynini.union(graph_with_key, graph_without_key)
        delete_tokens = self.delete_tokens(graph)

        self.fst = delete_tokens.optimize()
