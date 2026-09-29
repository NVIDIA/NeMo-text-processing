# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_SPACE, GraphFst


class WordFst(GraphFst):
    """
    Finite state transducer for classifying plain tokens, that do not belong to any special class. This can be considered as the default class.
        e.g. 文字 -> tokens { name: "文字" }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)
        word = pynutil.insert("name: \"") + NEMO_NOT_SPACE + pynutil.insert("\"")
        self.fst = word.optimize()
