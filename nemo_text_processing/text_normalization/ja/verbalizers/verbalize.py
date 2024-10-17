# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.ja.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ja.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.ja.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ja.verbalizers.fraction import FractionFst
from nemo_text_processing.text_normalization.ja.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.ja.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.ja.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.ja.verbalizers.word import WordFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        date = DateFst(deterministic=deterministic)
        cardinal = CardinalFst(deterministic=deterministic)
        ordinal = OrdinalFst(deterministic=deterministic)
        decimal = DecimalFst(deterministic=deterministic)
        word = WordFst(deterministic=deterministic)
        fraction = FractionFst(deterministic=deterministic)

        # money = MoneyFst(decimal=decimal, deterministic=deterministic)
        # measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
        time = TimeFst(deterministic=deterministic)
        whitelist = WhiteListFst(deterministic=deterministic)

        graph = pynini.union(
            date.fst, cardinal.fst, ordinal.fst, decimal.fst, fraction.fst, word.fst, time.fst, whitelist.fst,
        )
        graph = pynini.closure(delete_space) + graph + pynini.closure(delete_space)

        self.fst = graph.optimize()
