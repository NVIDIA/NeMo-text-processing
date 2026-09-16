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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, delete_space


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing plain tokens, e.g.
        tokens { name: "வணக்கம்" } -> வணக்கம்
    """

    def __init__(self):
        super().__init__(name="word", kind="verbalize")

        # A value may itself be a U+0022 QUOTATION MARK token, so only the space is excluded.
        chars = pynini.closure(NEMO_CHAR - " ", 1)
        graph = pynutil.delete("name: \"") + chars + pynutil.delete("\"")
        # Multi-word values travel with U+00A0 NO-BREAK SPACE; write them with plain spaces.
        graph = graph @ pynini.cdrewrite(pynini.cross(" ", " "), "", "", NEMO_SIGMA)
        self.fst = (delete_space + graph + delete_space).optimize()
