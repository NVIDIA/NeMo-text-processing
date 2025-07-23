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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_DIGIT, NEMO_NOT_SPACE, GraphFst
from nemo_text_processing.text_normalization.vi.taggers.punctuation import PunctuationFst


class WordFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese words.
        e.g. ngày -> name: "ngày"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        # Get punctuation marks to exclude from words
        punct = PunctuationFst().graph
        default_graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punct.project("input")), 1)

        # Exclude money symbols and digits to prevent conflicts with specialized taggers
        # Following English pattern but adapted for Vietnamese currency symbols
        symbols_to_exclude = (
            pynini.union("$", "€", "₩", "£", "¥", "¢", "₫", "đ", "#", "%", ",") | NEMO_DIGIT
        ).optimize()
        graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, symbols_to_exclude), 1)
        graph |= default_graph

        word = pynutil.insert("name: \"") + graph + pynutil.insert("\"")
        self.fst = word.optimize()
