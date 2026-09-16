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

from nemo_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_ALPHA,
    NEMO_NOT_SPACE,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_TA_BLOCK
from nemo_text_processing.text_normalization.ta.taggers.punctuation import PunctuationFst

# Symbols a semiotic class owns, so the word class must not swallow them.
_CLASS_SYMBOLS = ["$", "€", "₩", "£", "¥", "#", "%"]


class WordFst(GraphFst):
    """
    Finite state transducer for classifying words, e.g.
        தமிழ் -> tokens { name: "தமிழ்" }

    A run of Tamil characters is preferred over the fallback that accepts any non-space
    characters, and a URL stays one token instead of splitting into punctuation marks.

    Args:
        punctuation: PunctuationFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, punctuation: PunctuationFst, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        punct = punctuation.graph_input
        default_graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punct), 1)
        symbols_to_exclude = (pynini.union(*_CLASS_SYMBOLS) | punct).optimize()

        graph = pynini.closure(pynini.difference(NEMO_TA_BLOCK, symbols_to_exclude), 1)
        graph = pynutil.add_weight(graph, MIN_NEG_WEIGHT) | default_graph

        url_body = pynini.closure(pynini.difference(NEMO_NOT_SPACE, pynini.accep("\"")), 1)
        url = (pynini.closure(NEMO_ALPHA, 1) + "://" + url_body) | ("www." + url_body)
        graph = pynutil.add_weight(url, MIN_NEG_WEIGHT) | graph

        # No space is introduced around punctuation inside a word.
        graph = pynini.closure(graph + pynini.closure(punct + graph, 0, 1))

        self.graph = convert_space(graph)
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
