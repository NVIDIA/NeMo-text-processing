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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, delete_space


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing plain words, e.g.
        tokens { name: "தமிழ்" } -> தமிழ்

    A mark following a word attaches to it without a space. Multi-word values travel with
    U+00A0 NO-BREAK SPACE and are spoken with plain spaces.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="verbalize", deterministic=deterministic)

        chars = pynini.closure(NEMO_CHAR - " ", 1)
        punct = pynini.union("!", "?", ".", ",", "-", ":", ";", "।")
        char = pynutil.delete("name:") + delete_space + pynutil.delete("\"") + chars + pynutil.delete("\"")

        graph = char + pynini.closure(delete_space + punct, 0, 1)
        graph = graph @ pynini.cdrewrite(pynini.cross(" ", ""), "", punct, NEMO_SIGMA)
        graph = graph @ pynini.cdrewrite(pynini.cross(" ", " "), "", "", NEMO_SIGMA)

        self.fst = graph.optimize()
