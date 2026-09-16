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
from nemo_text_processing.inverse_text_normalization.ta.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.graph_utils import MIN_NEG_WEIGHT, NEMO_NOT_SPACE, convert_space
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_TA_BLOCK

# Symbols a semiotic class owns, so the word class must not swallow them.
_CLASS_SYMBOLS = ["$", "€", "₩", "£", "¥", "#", "%"]


class WordFst(GraphFst):
    """
    Finite state transducer for classifying plain tokens, that do not belong to any special class. This can be considered as the default class.
        e.g. வணக்கம் -> tokens { name: "வணக்கம்" }

    Args:
        punctuation: PunctuationFst
    """

    def __init__(self, punctuation: PunctuationFst):
        super().__init__(name="word", kind="classify")

        punct = punctuation.graph_input
        default_graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punct), 1)
        symbols_to_exclude = (pynini.union(*_CLASS_SYMBOLS) | punct).optimize()

        graph = pynini.closure(pynini.difference(NEMO_TA_BLOCK, symbols_to_exclude), 1)
        graph = pynutil.add_weight(graph, MIN_NEG_WEIGHT) | default_graph

        # No space is introduced around punctuation inside a word.
        graph = pynini.closure(graph + pynini.closure(punct + graph, 0, 1))

        self.graph = convert_space(graph)
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
