# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ko.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ko.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ko.verbalizers.fraction import FractionFst
from nemo_text_processing.text_normalization.ko.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.ko.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.ko.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.ko.verbalizers.whitelist import WhiteListFst





class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        cardinal = CardinalFst(deterministic=deterministic)
        date = DateFst(deterministic=deterministic)
        ordinal = OrdinalFst(deterministic=deterministic)
        decimal = DecimalFst(deterministic=deterministic)
        word = WordFst(deterministic=deterministic)
        fraction = FractionFst(deterministic=deterministic)
        whitelist = WhiteListFst(deterministic=deterministic)

        graph = pynini.union(cardinal.fst, ordinal.fst, word.fst, decimal.fst, fraction.fst, date.fst, whitelist.fst,)

        self.fst = graph.optimize()
