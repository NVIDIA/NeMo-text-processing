# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.bantu.graph_utils import run, word_token
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst


class WordFst(GraphFst):
    """
    Finite state transducer for classifying words left unchanged, e.g.
        kasa -> tokens { name: "kasa" }

    A run of number words that cannot be read as a number in its context also stays
    unchanged as a whole, word by word; it is never partly converted. Both carry a
    weight, so any valid number reading is preferred.

    Args:
        number_word: any word that can be part of a number
        ordinary_word: any other word, including a join word used as an ordinary word
    """

    def __init__(self, number_word: pynini.Fst, ordinary_word: pynini.Fst):
        super().__init__(name="word", kind="classify")

        unread_run = run(word_token(number_word) + pynini.closure(NEMO_SPACE + word_token(number_word)))
        graph = word_token(ordinary_word) | unread_run
        self.fst = pynutil.add_weight(graph, 1).optimize()
