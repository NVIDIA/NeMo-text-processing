# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.it.taggers.cardinal import CardinalFst as CardinalTagger
from nemo_text_processing.text_normalization.it.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.it.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.it.verbalizers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.it.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.it.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.it.verbalizers.time import TimeFst


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
        cardinal_graph = cardinal.fst
        decimal = DecimalFst(deterministic=deterministic)
        decimal_graph = decimal.fst
        electronic = ElectronicFst(deterministic=deterministic)
        electronic_graph = electronic.fst
        whitelist_graph = WhiteListFst(deterministic=deterministic).fst
        measure = MeasureFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
        measure_graph = measure.fst
        money = MoneyFst(decimal=decimal, deterministic=deterministic)
        money_graph = money.fst
        cardinal_tagger = CardinalTagger(deterministic=deterministic)
        time = TimeFst(cardinal_tagger=cardinal_tagger, deterministic=deterministic)
        time_graph = time.fst

        graph = (
            cardinal_graph
            | decimal_graph
            | electronic_graph
            | whitelist_graph
            | measure_graph
            | money_graph
            | time_graph
        )

        self.fst = graph
