# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.el.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.el.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.el.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.el.verbalizers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.el.verbalizers.fraction import FractionFst
from nemo_text_processing.text_normalization.el.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.el.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.el.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.el.verbalizers.roman import RomanFst
from nemo_text_processing.text_normalization.el.verbalizers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.el.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.el.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars. For deployment, this grammar will be compiled and
    exported to OpenFst Finite State Archive (FAR) File.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal_graph = CardinalFst(deterministic=deterministic).fst
        ordinal_graph = OrdinalFst(deterministic=deterministic).fst
        decimal_graph = DecimalFst(deterministic=deterministic).fst
        fraction_graph = FractionFst(deterministic=deterministic).fst
        money_graph = MoneyFst(deterministic=deterministic).fst
        date_graph = DateFst(deterministic=deterministic).fst
        time_graph = TimeFst(deterministic=deterministic).fst
        measure_graph = MeasureFst(deterministic=deterministic).fst
        telephone_graph = TelephoneFst(deterministic=deterministic).fst
        electronic_graph = ElectronicFst(deterministic=deterministic).fst
        whitelist_graph = WhiteListFst().fst
        roman_graph = RomanFst(deterministic=deterministic).fst
        graph = (
            cardinal_graph
            | ordinal_graph
            | decimal_graph
            | fraction_graph
            | money_graph
            | date_graph
            | time_graph
            | measure_graph
            | telephone_graph
            | electronic_graph
            | whitelist_graph
            | roman_graph
        )
        self.fst = graph.optimize()
