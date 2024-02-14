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

from nemo_text_processing.inverse_text_normalization.ar.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.ar.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.ar.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.ar.taggers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.ar.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.ar.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.ar.taggers.word import WordFst
from nemo_text_processing.text_normalization.ar.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.ar.taggers.tokenize_and_classify import ClassifyFst as TNClassifyFst
from nemo_text_processing.text_normalization.en.graph_utils import INPUT_LOWER_CASED
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
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"ar_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars.")
            tn_classify = TNClassifyFst(
                input_case='cased', deterministic=True, cache_dir=cache_dir, overwrite_cache=True
            )

            cardinal = CardinalFst(tn_cardinal=tn_classify.cardinal)
            cardinal_graph = cardinal.fst
            decimal = DecimalFst(tn_decimal=tn_classify.decimal)
            decimal_graph = decimal.fst
            fraction = FractionFst(tn_cardinal=tn_classify.cardinal)
            fraction_graph = fraction.fst
            money = MoneyFst(itn_cardinal_tagger=cardinal)
            money_graph = money.fst
            measure = MeasureFst(
                itn_cardinal_tagger=cardinal,
                itn_decimal_tagger=decimal,
                itn_fraction_tagger=fraction,
                deterministic=True,
            )
            measure_graph = measure.fst
            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst

            classify = (
                pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
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
