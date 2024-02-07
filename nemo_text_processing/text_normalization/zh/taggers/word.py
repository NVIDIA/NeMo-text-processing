# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_CHAR, NEMO_NOT_QUOTE, NEMO_NOT_SPACE, GraphFst
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word.
        e.g. dormir -> tokens { name: "dormir" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify")
        #word = pynutil.insert("name: \"") + NEMO_NOT_QUOTE + pynutil.insert("\"") # original line
        word = pynutil.insert("name: \"") + NEMO_NOT_QUOTE + pynutil.insert("\"")
        self.fst = word.optimize()
        #import pdb; pdb.set_trace() 
