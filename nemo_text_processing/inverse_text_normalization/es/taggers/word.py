# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_SPACE, GraphFst
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for classifying plain tokens, that do not belong to any special class. This can be considered as the default class.
        e.g. sleep -> tokens { name: "sleep" }
    """

    def __init__(self):
        super().__init__(name="word", kind="classify")
        word = pynutil.insert("name: \"") + pynini.closure(NEMO_NOT_SPACE, 1) + pynutil.insert("\"")
        self.fst = word.optimize()
