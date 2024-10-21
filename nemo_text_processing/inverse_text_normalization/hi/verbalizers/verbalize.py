# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        fraction_graph = FractionFst().fst
        date_graph = DateFst().fst
        time_graph = TimeFst().fst
        measure_graph = MeasureFst(cardinal, decimal).fst
        money_graph = MoneyFst(cardinal, decimal).fst

        whitelist_graph = WhiteListFst().fst

        graph = (
            cardinal_graph
            | whitelist_graph
            | ordinal_graph
            | decimal_graph
            | fraction_graph
            | date_graph
            | time_graph
            | measure_graph
            | money_graph
        )
        self.fst = graph
