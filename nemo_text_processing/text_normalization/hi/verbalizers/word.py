# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, delete_space


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing Hindi words.
        e.g. tokens { name: "सोना" } -> सोना

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="verbalize", deterministic=deterministic)
        chars = pynini.closure(NEMO_CHAR - " ", 1)
        punct = pynini.union("!", "?", ".", ",", "-", ":", ";", "।")  # Add other punctuation marks as needed
        char = pynutil.delete("name:") + delete_space + pynutil.delete("\"") + chars + pynutil.delete("\"")

        # Ensure no spaces around punctuation
        graph = char + pynini.closure(delete_space + punct, 0, 1)

        # Explicitly remove spaces before punctuation
        remove_space_before_punct = pynini.cdrewrite(pynini.cross(" ", ""), "", punct, NEMO_SIGMA)
        graph = graph @ remove_space_before_punct

        self.fst = graph.optimize()
