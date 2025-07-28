# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hy.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.hy.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        fraction = FractionFst()
        fraction_graph = fraction.fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal).fst
        money_graph = MoneyFst(decimal=decimal).fst
        time_graph = TimeFst().fst
        whitelist_graph = WhiteListFst().fst
        graph = (
            time_graph
            | measure_graph
            | fraction_graph
            | money_graph
            | ordinal_graph
            | decimal_graph
            | cardinal_graph
            | whitelist_graph
        )
        self.fst = graph
