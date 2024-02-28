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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.en.taggers.cardinal import CardinalFst as EnCardinalFst
from nemo_text_processing.inverse_text_normalization.en.taggers.date import DateFst as EnDateFst
from nemo_text_processing.inverse_text_normalization.en.taggers.decimal import DecimalFst as EnDecimalFst
from nemo_text_processing.inverse_text_normalization.en.taggers.electronic import ElectronicFst as EnElectronicFst
from nemo_text_processing.inverse_text_normalization.en.taggers.measure import MeasureFst as EnMeasureFst
from nemo_text_processing.inverse_text_normalization.en.taggers.money import MoneyFst as EnMoneyFst
from nemo_text_processing.inverse_text_normalization.en.taggers.ordinal import OrdinalFst as EnOrdinalFst
from nemo_text_processing.inverse_text_normalization.en.taggers.punctuation import PunctuationFst as EnPunctuationFst
from nemo_text_processing.inverse_text_normalization.en.taggers.telephone import TelephoneFst as EnTelephoneFst
from nemo_text_processing.inverse_text_normalization.en.taggers.time import TimeFst as EnTimeFst
from nemo_text_processing.inverse_text_normalization.en.taggers.whitelist import WhiteListFst as EnWhiteListFst
from nemo_text_processing.inverse_text_normalization.en.taggers.word import WordFst as EnWordFst
from nemo_text_processing.inverse_text_normalization.es.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.es.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.es.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.es.taggers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.es.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.es.taggers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.es.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.es.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.es.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.es.taggers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.es.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.es.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.es.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.es_en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        en_whitelist: str = None,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if whitelist is None:
            whitelist = get_abs_path("data/es_whitelist.tsv")
        if en_whitelist is None:
            en_whitelist = get_abs_path("data/en_whitelist.tsv")
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"es_en_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars.")

            cardinal = CardinalFst(input_case=input_case)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal, input_case=input_case)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal, input_case=input_case)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal, ordinal, input_case=input_case)
            fraction_graph = fraction.fst

            measure_graph = MeasureFst(
                cardinal=cardinal, decimal=decimal, fraction=fraction, input_case=input_case
            ).fst
            date_graph = DateFst(cardinal, input_case=input_case).fst
            word_graph = WordFst().fst
            time_graph = TimeFst(input_case=input_case).fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, input_case=input_case).fst
            whitelist_graph = WhiteListFst(input_file=whitelist).fst
            punct_graph = PunctuationFst().fst
            electronic_graph = ElectronicFst(input_case=input_case).fst
            telephone_graph = TelephoneFst(input_case=input_case).fst

            en_cardinal = EnCardinalFst(input_case=input_case)
            en_cardinal_graph = en_cardinal.fst

            en_ordinal = EnOrdinalFst(cardinal=en_cardinal, input_case=input_case)
            en_ordinal_graph = en_ordinal.fst

            en_decimal = EnDecimalFst(cardinal=en_cardinal, input_case=input_case)
            en_decimal_graph = en_decimal.fst

            en_measure_graph = EnMeasureFst(cardinal=en_cardinal, decimal=en_decimal, input_case=input_case).fst
            en_date_graph = EnDateFst(ordinal=en_ordinal, input_case=input_case).fst
            en_word_graph = EnWordFst().fst
            en_time_graph = EnTimeFst(input_case=input_case).fst
            en_money_graph = EnMoneyFst(cardinal=en_cardinal, decimal=en_decimal, input_case=input_case).fst
            en_whitelist_graph = EnWhiteListFst(input_file=en_whitelist, input_case=input_case).fst
            en_punct_graph = EnPunctuationFst().fst
            en_electronic_graph = EnElectronicFst(input_case=input_case).fst
            en_telephone_graph = EnTelephoneFst(cardinal=en_cardinal, input_case=input_case).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(en_whitelist_graph, 1.01)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(en_time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(en_date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 1.09)
                | pynutil.add_weight(en_decimal_graph, 1.09)
                | pynutil.add_weight(fraction_graph, 1.09)
                | pynutil.add_weight(measure_graph, 1.6)
                | pynutil.add_weight(en_measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.6)
                | pynutil.add_weight(en_cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.6)
                | pynutil.add_weight(en_ordinal_graph, 1.09)
                | pynutil.add_weight(money_graph, 1.6)
                | pynutil.add_weight(en_money_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.6)
                | pynutil.add_weight(en_telephone_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 2.3)
                | pynutil.add_weight(en_electronic_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
                | pynutil.add_weight(en_word_graph, 120)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            en_punct = (
                pynutil.insert("tokens { ") + pynutil.add_weight(en_punct_graph, weight=1.3) + pynutil.insert(" }")
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + token
                + pynini.closure(pynutil.insert(" ") + punct | en_punct)
            )

            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logger.info(f"ClassifyFst grammars are saved to {far_file}.")
