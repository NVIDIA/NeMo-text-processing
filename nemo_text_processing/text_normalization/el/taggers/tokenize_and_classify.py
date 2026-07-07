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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.el.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.el.taggers.date import DateFst
from nemo_text_processing.text_normalization.el.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.el.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.el.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.el.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.el.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.el.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.el.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.el.taggers.range import RangeFst
from nemo_text_processing.text_normalization.el.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.el.taggers.serial import SerialFst
from nemo_text_processing.text_normalization.el.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.el.taggers.time import TimeFst
from nemo_text_processing.text_normalization.el.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.el.taggers.word import WordFst
from nemo_text_processing.text_normalization.el.verbalizers.date import DateFst as vDateFst
from nemo_text_processing.text_normalization.el.verbalizers.time import TimeFst as vTimeFst
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
    Final class that composes all other classification grammars. This class can process an
    entire sentence. For deployment, this grammar will be compiled and exported to OpenFst
    Finite State Archive (FAR) File.

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
        input_case: str = INPUT_LOWER_CASED,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"_el_tn_{input_case}_{deterministic}_deterministic.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info("Creating ClassifyFst grammars.")

            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic)
            fraction_graph = fraction.fst

            date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst
            time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            measure_graph = MeasureFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic).fst
            telephone_graph = TelephoneFst(cardinal=cardinal, deterministic=deterministic).fst
            electronic_graph = ElectronicFst(deterministic=deterministic).fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic).fst
            serial_graph = SerialFst(cardinal=cardinal, deterministic=deterministic).fst

            v_time_fst = vTimeFst(deterministic=deterministic).fst
            v_date_fst = vDateFst(deterministic=deterministic).fst
            time_final = pynini.compose(time_graph, v_time_fst)
            date_final = pynini.compose(date_graph, v_date_fst)
            range_graph = RangeFst(
                time=time_final, date=date_final, cardinal=cardinal, deterministic=deterministic
            ).fst

            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst
            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst
            roman_graph = RomanFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(date_graph, 1.08)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.2)
                | pynutil.add_weight(measure_graph, 1.2)
                | pynutil.add_weight(roman_graph, 1.2)
                | pynutil.add_weight(telephone_graph, 1.3)
                | pynutil.add_weight(serial_graph, 1.3)
                | pynutil.add_weight(electronic_graph, 1.3)
                | pynutil.add_weight(word_graph, 100)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logger.info(f"ClassifyFst grammars are saved to {far_file}.")
