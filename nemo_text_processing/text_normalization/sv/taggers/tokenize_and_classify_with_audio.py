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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import AbbreviationFst as vAbbreviationFst
from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst as vWordFst
from nemo_text_processing.text_normalization.sv.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.sv.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.sv.taggers.date import DateFst
from nemo_text_processing.text_normalization.sv.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.sv.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.sv.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.sv.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.sv.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.sv.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.sv.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.sv.taggers.time import TimeFst
from nemo_text_processing.text_normalization.sv.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.sv.taggers.word import WordFst
from nemo_text_processing.text_normalization.sv.verbalizers.cardinal import CardinalFst as vCardinalFst
from nemo_text_processing.text_normalization.sv.verbalizers.date import DateFst as vDateFst
from nemo_text_processing.text_normalization.sv.verbalizers.decimals import DecimalFst as vDecimalFst
from nemo_text_processing.text_normalization.sv.verbalizers.electronic import ElectronicFst as vElectronicFst
from nemo_text_processing.text_normalization.sv.verbalizers.fraction import FractionFst as vFractionFst
from nemo_text_processing.text_normalization.sv.verbalizers.measure import MeasureFst as vMeasureFst
from nemo_text_processing.text_normalization.sv.verbalizers.money import MoneyFst as vMoneyFst
from nemo_text_processing.text_normalization.sv.verbalizers.ordinal import OrdinalFst as vOrdinalFst
from nemo_text_processing.text_normalization.sv.verbalizers.telephone import TelephoneFst as vTelephoneFst
from nemo_text_processing.text_normalization.sv.verbalizers.time import TimeFst as vTimeFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = True,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_sv_tn_{deterministic}_deterministic_{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f'ClassifyFst.fst was restored from {far_file}.')
        else:
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            fraction = FractionFst(deterministic=deterministic, ordinal=ordinal, cardinal=cardinal)
            fraction_graph = fraction.fst
            measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            measure_graph = measure.fst
            date_graph = DateFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic).fst
            punctuation = PunctuationFst(deterministic=True)
            punct_graph = punctuation.graph
            time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            word_graph = WordFst(deterministic=deterministic).graph
            telephone_graph = TelephoneFst(deterministic=deterministic).fst
            electronic_graph = ElectronicFst(deterministic=deterministic).fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic).fst
            whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist)
            whitelist_graph = whitelist.graph

            v_cardinal = vCardinalFst(deterministic=deterministic)
            v_cardinal_graph = v_cardinal.fst
            v_decimal = vDecimalFst(deterministic=deterministic)
            v_decimal_graph = v_decimal.fst
            v_ordinal = vOrdinalFst(deterministic=deterministic)
            v_ordinal_graph = v_ordinal.fst
            v_fraction = vFractionFst(deterministic=deterministic)
            v_fraction_graph = v_fraction.fst
            v_telephone_graph = vTelephoneFst(deterministic=deterministic).fst
            v_electronic_graph = vElectronicFst(deterministic=deterministic).fst
            v_measure = vMeasureFst(decimal=decimal, cardinal=cardinal, fraction=fraction, deterministic=deterministic)
            v_measure_graph = v_measure.fst
            v_time_graph = vTimeFst(deterministic=deterministic).fst
            v_date_graph = vDateFst(deterministic=deterministic).fst
            v_money_graph = vMoneyFst(decimal=decimal, deterministic=deterministic).fst
            v_abbreviation = vAbbreviationFst(deterministic=deterministic).fst

            v_word_graph = vWordFst(deterministic=deterministic).fst

            sem_w = 1
            word_w = 100
            punct_w = 2
            classify_and_verbalize = (
                pynutil.add_weight(whitelist_graph, sem_w)
                | pynutil.add_weight(pynini.compose(time_graph, v_time_graph), sem_w)
                | pynutil.add_weight(pynini.compose(decimal_graph, v_decimal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(measure_graph, v_measure_graph), sem_w)
                | pynutil.add_weight(pynini.compose(cardinal_graph, v_cardinal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(ordinal_graph, v_ordinal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(telephone_graph, v_telephone_graph), sem_w)
                | pynutil.add_weight(pynini.compose(electronic_graph, v_electronic_graph), sem_w)
                | pynutil.add_weight(pynini.compose(fraction_graph, v_fraction_graph), sem_w)
                | pynutil.add_weight(pynini.compose(money_graph, v_money_graph), sem_w)
                | pynutil.add_weight(word_graph, word_w)
                | pynutil.add_weight(pynini.compose(date_graph, v_date_graph), sem_w - 0.01)
                | pynutil.add_weight(v_word_graph, 1.1001)
            ).optimize()

            if not deterministic:
                abbreviation_graph = AbbreviationFst(whitelist=whitelist, deterministic=deterministic).fst
                classify_and_verbalize |= pynutil.add_weight(
                    pynini.compose(abbreviation_graph, v_abbreviation), word_w
                )

            punct_only = pynutil.add_weight(punct_graph, weight=punct_w)
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct_only),
                1,
            )

            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + classify_and_verbalize
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph |= punct_only + pynini.closure(punct)
            graph = delete_space + graph + delete_space

            remove_extra_spaces = pynini.closure(NEMO_NOT_SPACE, 1) + pynini.closure(
                delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1)
            )
            remove_extra_spaces |= (
                pynini.closure(pynutil.delete(" "), 1)
                + pynini.closure(NEMO_NOT_SPACE, 1)
                + pynini.closure(delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1))
            )

            graph = pynini.compose(graph.optimize(), remove_extra_spaces).optimize()
            self.fst = graph

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})

        # to remove normalization options that still contain digits and some special symbols
        # e.g., "P&E" -> {P and E, P&E}, "P & E" will be removed from the list of normalization options
        no_digits = pynini.closure(pynini.difference(NEMO_CHAR, pynini.union(NEMO_DIGIT, "&")))
        self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
