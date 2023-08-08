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

import logging
import os

import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.zh.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.zh.taggers.date import DateFst
from nemo_text_processing.text_normalization.zh.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.zh.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.zh.taggers.math_symbol import MathSymbol
from nemo_text_processing.text_normalization.zh.taggers.measure import Measure
from nemo_text_processing.text_normalization.zh.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.zh.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.zh.taggers.preprocessor import PreProcessor
from nemo_text_processing.text_normalization.zh.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.zh.taggers.time import TimeFst
from nemo_text_processing.text_normalization.zh.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.zh.taggers.word import Char
from pynini.lib import pynutil


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
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
        deterministic: bool = False,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                # cache_dir, f"_{input_case}_zh_tn_{deterministic}_deterministic{whitelist_file}.far"
                cache_dir,
                f"_{input_case}_zh_tn_{deterministic}_deterministic_{whitelist_file}.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            no_digits = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars. This might take some time...")

            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal=cardinal)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            fraction_graph = fraction.fst

            date = DateFst(deterministic=deterministic)
            date_graph = date.fst

            word_graph = Char(deterministic=deterministic).fst

            self.time = TimeFst(deterministic=deterministic)
            time_graph = self.time.fst

            money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            money_graph = money.fst

            self.math = MathSymbol(deterministic=deterministic)
            math_graph = self.math.fst

            self.measure = Measure(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            measure_graph = self.measure.fst

            self.whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist)
            whitelist_graph = self.whitelist.fst
            punct_graph = PunctuationFst(deterministic=deterministic).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.001)
                | pynutil.add_weight(cardinal_graph, -2.0)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(fraction_graph, -1.1)
                | pynutil.add_weight(date_graph, -1.0)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, -1.0)
                | pynutil.add_weight(money_graph, -1.1)
                | pynutil.add_weight(math_graph, 1.1)
                | pynutil.add_weight(measure_graph, -1.1)
                | pynutil.add_weight(word_graph, 1.1)
            )

            classify |= pynutil.add_weight(word_graph, 100)

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct)
            graph = delete_space + graph + delete_space

            # self.fst = graph.optimize()
            tagger = graph.optimize()
            preprocessor = PreProcessor(remove_interjections=True, fullwidth_to_halfwidth=True,)
            self.fst = preprocessor.fst @ tagger

            no_digits = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
