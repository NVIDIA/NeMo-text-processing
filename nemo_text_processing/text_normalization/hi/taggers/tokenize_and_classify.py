# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import time

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
    generate_far_filename,
)
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hi.taggers.date import DateFst
from nemo_text_processing.text_normalization.hi.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.hi.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.hi.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.hi.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.hi.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.hi.taggers.time import TimeFst
from nemo_text_processing.text_normalization.hi.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.hi.taggers.word import WordFst


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
        project_input: bool = False,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = generate_far_filename(
                language="hi",
                mode="tn",
                cache_dir=cache_dir,
                operation="tokenize",
                deterministic=deterministic,
                project_input=project_input,
                input_case=input_case,
                whitelist_file=whitelist_file
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")

            start_time = time.time()
            cardinal = CardinalFst(deterministic=deterministic, project_input=project_input)
            cardinal_graph = cardinal.fst
            logging.debug(f"cardinal: {time.time() - start_time: .2f}s -- {cardinal_graph.num_states()} nodes")

            start_time = time.time()
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic, project_input=project_input)
            decimal_graph = decimal.fst
            logging.debug(f"decimal: {time.time() - start_time: .2f}s -- {decimal_graph.num_states()} nodes")

            start_time = time.time()
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic, project_input=project_input)
            fraction_graph = fraction.fst
            logging.debug(f"fraction: {time.time() - start_time: .2f}s -- {fraction_graph.num_states()} nodes")

            start_time = time.time()
            date = DateFst(cardinal=cardinal, project_input=project_input)
            date_graph = date.fst
            logging.debug(f"date: {time.time() - start_time: .2f}s -- {date_graph.num_states()} nodes")

            start_time = time.time()
            timefst = TimeFst(project_input=project_input)
            time_graph = timefst.fst
            logging.debug(f"time: {time.time() - start_time: .2f}s -- {time_graph.num_states()} nodes")

            start_time = time.time()
            measure = MeasureFst(cardinal=cardinal, decimal=decimal, project_input=project_input)
            measure_graph = measure.fst
            logging.debug(f"measure: {time.time() - start_time: .2f}s -- {measure_graph.num_states()} nodes")

            start_time = time.time()
            money = MoneyFst(cardinal=cardinal, decimal=decimal, project_input=project_input)
            money_graph = money.fst
            logging.debug(f"money: {time.time() - start_time: .2f}s -- {money_graph.num_states()} nodes")

            start_time = time.time()
            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist, project_input=project_input
            ).fst
            logging.debug(f"whitelist: {time.time() - start_time: .2f}s -- {whitelist_graph.num_states()} nodes")

            start_time = time.time()
            punctuation = PunctuationFst(deterministic=deterministic, project_input=project_input)
            punct_graph = punctuation.fst
            logging.debug(f"punct: {time.time() - start_time: .2f}s -- {punct_graph.num_states()} nodes")

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
            )

            start_time = time.time()
            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic, project_input=project_input).fst
            logging.debug(f"word: {time.time() - start_time: .2f}s -- {word_graph.num_states()} nodes")

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=2.1) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct),
                1,
            )

            classify |= pynutil.add_weight(word_graph, 100)
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph |= punct

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
