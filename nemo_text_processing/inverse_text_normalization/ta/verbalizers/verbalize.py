# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License.import pynini
from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.ta.verbalizers.word import WordFst


class VerbalizeFst(GraphFst):
    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        cardinal = CardinalFst()
        word = WordFst()

        self.fst = (cardinal.fst | word.fst).optimize()
