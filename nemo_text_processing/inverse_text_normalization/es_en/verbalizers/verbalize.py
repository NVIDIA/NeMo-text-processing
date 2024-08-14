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

from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.en.verbalizers.cardinal import CardinalFst as EnCardinalFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.date import DateFst as EnDateFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.decimal import DecimalFst as EnDecimalFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.electronic import ElectronicFst as EnElectronicFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.measure import MeasureFst as EnMeasureFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.money import MoneyFst as EnMoneyFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.ordinal import OrdinalFst as EnOrdinalFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.telephone import TelephoneFst as EnTelephoneFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.time import TimeFst as EnTimeFst
from nemo_text_processing.inverse_text_normalization.en.verbalizers.whitelist import WhiteListFst as EnWhiteListFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.es.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


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

        decimal = DecimalFst()
        decimal_graph = decimal.fst

        fraction = FractionFst()
        fraction_graph = fraction.fst

        ordinal_graph = OrdinalFst().fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal, fraction=fraction).fst
        money_graph = MoneyFst(decimal=decimal).fst
        time_graph = TimeFst().fst
        date_graph = DateFst().fst
        whitelist_graph = WhiteListFst().fst
        telephone_graph = TelephoneFst().fst
        electronic_graph = ElectronicFst().fst

        en_cardinal = EnCardinalFst()
        en_cardinal_graph = en_cardinal.fst
        en_ordinal_graph = EnOrdinalFst().fst
        en_decimal = EnDecimalFst()
        en_decimal_graph = en_decimal.fst
        en_measure_graph = EnMeasureFst(decimal=en_decimal, cardinal=en_cardinal).fst
        en_money_graph = EnMoneyFst(decimal=en_decimal).fst
        en_date_graph = EnDateFst().fst
        en_whitelist_graph = EnWhiteListFst().fst
        en_telephone_graph = EnTelephoneFst().fst
        en_time_graph = EnTimeFst().fst
        en_electronic_graph = EnElectronicFst().fst

        graph = (
            en_time_graph
            | pynutil.add_weight(time_graph, 1.1)
            | date_graph
            | pynutil.add_weight(en_date_graph, 1.1)
            | money_graph
            | pynutil.add_weight(en_money_graph, 1.1)
            | fraction_graph
            | measure_graph
            | pynutil.add_weight(en_measure_graph, 1.1)
            | ordinal_graph
            | pynutil.add_weight(en_ordinal_graph, 1.1)
            | decimal_graph
            | pynutil.add_weight(en_decimal_graph, 1.1)
            | cardinal_graph
            | pynutil.add_weight(en_cardinal_graph, 1.1)
            | whitelist_graph
            | pynutil.add_weight(en_whitelist_graph, 1.1)
            | telephone_graph
            | pynutil.add_weight(en_telephone_graph, 1.1)
            | electronic_graph
            | pynutil.add_weight(en_electronic_graph, 1.1)
        )
        self.fst = graph
