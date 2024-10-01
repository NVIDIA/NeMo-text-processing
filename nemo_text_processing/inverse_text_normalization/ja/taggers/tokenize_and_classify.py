# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    GraphFst,
    generator_main,
)
from nemo_text_processing.inverse_text_normalization.ja.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.ja.taggers.word import WordFst


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str = INPUT_LOWER_CASED,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"jp_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal)
            ordinal_graph = ordinal.fst

            date = DateFst(cardinal)
            date_graph = date.fst

            decimal = DecimalFst(cardinal)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal, decimal)
            fraction_graph = fraction.fst

            time = TimeFst()
            time_graph = time.fst

            word_graph = WordFst().fst
            whitelist_graph = WhiteListFst().fst
            punct_graph = PunctuationFst().fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(cardinal_graph, 1.0)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.0)
                | pynutil.add_weight(time_graph, 1.0)
                | pynutil.add_weight(word_graph, 100)
                | pynutil.add_weight(punct_graph, 1.1)
            )

            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")
            tagger = pynini.closure(token, 1)

            self.fst = tagger

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
