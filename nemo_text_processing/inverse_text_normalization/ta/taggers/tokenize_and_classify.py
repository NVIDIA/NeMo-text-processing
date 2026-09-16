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

import logging
import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst, generator_main
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.word import WordFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.ta.taggers.cardinal import CardinalFst as TnCardinalFst


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    The spoken number forms are the Tamil TN cardinal's own grammar inverted, so the two
    directions share one description of the number morphology.

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
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(cache_dir, f"ta_itn_{input_case}_{whitelist_file}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst(TnCardinalFst())
            cardinal_graph = cardinal.fst
            decimal_graph = DecimalFst(cardinal).fst
            fraction_graph = FractionFst(cardinal).fst
            ordinal_graph = OrdinalFst(cardinal).fst
            date_graph = DateFst(cardinal).fst
            time_graph = TimeFst(cardinal).fst
            money_graph = MoneyFst(cardinal).fst
            telephone_graph = TelephoneFst(cardinal).fst
            whitelist_graph = WhiteListFst(input_file=whitelist).fst
            punctuation = PunctuationFst()
            punct_graph = punctuation.fst
            word_graph = WordFst(punctuation).fst

            # A written number passes through (whitelist, below 1.0), then the classes from the
            # most to the least specific reading of a spoken number.
            classify = (
                pynutil.add_weight(whitelist_graph, 1.0)
                | pynutil.add_weight(telephone_graph, 0.9)
                | pynutil.add_weight(date_graph, 1.04)
                | pynutil.add_weight(time_graph, 1.05)
                | pynutil.add_weight(fraction_graph, 1.06)
                | pynutil.add_weight(money_graph, 1.07)
                | pynutil.add_weight(decimal_graph, 1.08)
                | pynutil.add_weight(ordinal_graph, 1.09)
                | pynutil.add_weight(cardinal_graph, 1.1)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=2.1) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.union(
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space),
                    (pynutil.insert(NEMO_SPACE) + punct),
                ),
                1,
            )

            classify = pynini.union(classify, pynutil.add_weight(word_graph, 100))
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(NEMO_SPACE))
                + token
                + pynini.closure(pynutil.insert(NEMO_SPACE) + punct)
            )

            graph = token_plus_punct + pynini.closure(
                pynini.union(
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space),
                    (pynutil.insert(NEMO_SPACE) + punct + pynutil.insert(NEMO_SPACE)),
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph = pynini.union(graph, punct)

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
