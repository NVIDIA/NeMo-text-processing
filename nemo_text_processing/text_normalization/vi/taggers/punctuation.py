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

from nemo_text_processing.text_normalization.vi.graph_utils import GraphFst


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation for Vietnamese
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="punctuation", kind="classify", deterministic=deterministic)

        # Common punctuation marks (excluding symbols that should be handled by other taggers)
        s = "!#%&'()*+-./:;<=>?@^_`{|}~"
        
        # Exclude symbols that should be handled by specialized taggers
        symbols_to_exclude = ["$", "€", "₩", "£", "¥", "¢", "₫", "đ", ","]  # comma is handled by money/decimal
        
        # Filter out excluded symbols
        punct_marks = [p for p in s if p not in symbols_to_exclude]
        self.punct_marks = punct_marks
        
        punct = pynini.union(*punct_marks)

        # Create the punctuation transduction
        graph = pynutil.insert('name: "') + punct + pynutil.insert('"')
        self.graph = punct

        final_graph = pynutil.insert("punctuation { ") + graph + pynutil.insert(" }")
        self.fst = final_graph.optimize()
