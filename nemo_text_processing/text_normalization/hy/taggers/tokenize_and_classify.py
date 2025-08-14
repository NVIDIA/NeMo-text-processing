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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generate_far_filename,
    generator_main,
)
from nemo_text_processing.text_normalization.hy.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hy.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.hy.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.hy.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.hy.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.hy.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.hy.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.hy.taggers.time import TimeFst
from nemo_text_processing.text_normalization.hy.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.hy.taggers.word import WordFst
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
        cache_dir: str = None,
        whitelist: str = None,
        deterministic: bool = False,
        project_input: bool = False,
        overwrite_cache: bool = False,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = generate_far_filename(
                language="hy",
                mode="tn",
                cache_dir=cache_dir,
                operation="tokenize",
                deterministic=deterministic,
                project_input=project_input,
                input_case=input_case,
                whitelist_file="",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars.")

            cardinal = CardinalFst(project_input=project_input)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal, project_input=project_input)
            ordinal_graph = ordinal.fst

            fraction = FractionFst(cardinal=cardinal, ordinal=ordinal, project_input=project_input)
            fraction_graph = fraction.fst

            decimal = DecimalFst(cardinal, project_input=project_input)
            decimal_graph = decimal.fst

            measure_graph = MeasureFst(cardinal=cardinal, decimal=decimal, project_input=project_input).fst
            word_graph = WordFst(project_input=project_input).fst
            time_graph = TimeFst(project_input=project_input).fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, project_input=project_input).fst
            punct_graph = PunctuationFst(project_input=project_input).fst
            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist, project_input=project_input
            ).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(measure_graph, 0.9)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.09)
                | pynutil.add_weight(money_graph, 1.1)
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
