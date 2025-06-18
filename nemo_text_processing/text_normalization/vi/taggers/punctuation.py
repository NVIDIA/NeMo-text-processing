# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation for Vietnamese
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="punctuation", kind="classify", deterministic=deterministic)

        # Common punctuation marks
        # Use escape() for brackets since they are special regex chars
        s = "!#$%&'()*+,-./:;<=>?@^_`{|}~–—――…»«„“›‹‚‘’⟨⟩"
        punct = pynini.union(*s)

        # Create the punctuation transduction
        graph = pynutil.insert('name: "') + punct + pynutil.insert('"')

        final_graph = pynutil.insert("punctuation { ") + graph + pynutil.insert(" }")
        self.fst = final_graph.optimize()
