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

import logging
import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.mr.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.inverse_text_normalization.mr.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.word import WordFst


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment

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
            far_file = os.path.join(cache_dir, f"mr_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst
            decimal_graph = DecimalFst(cardinal).fst
            time_graph = TimeFst().fst
            date_graph = DateFst(cardinal).fst

            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst
            classify = (
                pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
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
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
