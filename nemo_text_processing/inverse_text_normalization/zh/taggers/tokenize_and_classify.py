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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    delete_zero_or_one_space,
    generator_main,
)
from nemo_text_processing.inverse_text_normalization.zh.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.word import WordFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(
        self,
        input_case: str,
        cache_dir: str = None,
        whitelist: str = None,
        overwrite_cache: bool = False,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "_zh_itn.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal)
            decimal_graph = decimal.fst

            date_graph = DateFst().fst
            word_graph = WordFst().fst
            time_graph = TimeFst().fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal).fst
            fraction = FractionFst(cardinal)
            fraction_graph = fraction.fst
            punct_graph = PunctuationFst().fst
            whitelist_graph = WhiteListFst(input_file=whitelist, input_case=input_case).fst

            classify = (
                pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 1.2)
                | pynutil.add_weight(cardinal_graph, 1.09)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
                | pynutil.add_weight(whitelist_graph, 1.01)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(delete_zero_or_one_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
