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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.vi.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.vi.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        # Initialize verbalizers
        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst

        whitelist = WhiteListFst(deterministic=deterministic)
        whitelist_graph = whitelist.fst

        word = WordFst(deterministic=deterministic)
        word_graph = word.fst

        # Combine all verbalizers
        graph = cardinal_graph | whitelist_graph | word_graph

        self.fst = graph
