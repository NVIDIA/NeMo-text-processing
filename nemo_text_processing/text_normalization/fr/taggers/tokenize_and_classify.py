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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
    generate_far_filename,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.fr.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.fr.taggers.date import DateFst
from nemo_text_processing.text_normalization.fr.taggers.decimals import DecimalFst
from nemo_text_processing.text_normalization.fr.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.fr.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.fr.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.fr.taggers.word import WordFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State aRchive (FAR) File.
    More details to deployment at NeMo-text-processing/tools/text_processing_deployment.
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
                language="fr",
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
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars. This might take some time...")

            self.cardinal = CardinalFst(deterministic=deterministic, project_input=project_input)
            cardinal_graph = self.cardinal.fst

            self.ordinal = OrdinalFst(cardinal=self.cardinal, deterministic=deterministic, project_input=project_input)
            ordinal_graph = self.ordinal.fst

            self.decimal = DecimalFst(cardinal=self.cardinal, deterministic=deterministic, project_input=project_input)
            decimal_graph = self.decimal.fst

            self.fraction = FractionFst(cardinal=self.cardinal, ordinal=self.ordinal, deterministic=deterministic, project_input=project_input)
            fraction_graph = self.fraction.fst
            word_graph = WordFst(deterministic=deterministic, project_input=project_input).fst
            self.whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist, project_input=project_input)
            whitelist_graph = self.whitelist.fst
            punct_graph = PunctuationFst(deterministic=deterministic, project_input=project_input).fst

            self.date = DateFst(self.cardinal, deterministic=deterministic, project_input=project_input)
            date_graph = self.date.fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.09)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(word_graph, 200)
            )
            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=2.1) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct),
                1,
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure((delete_extra_space).ques + token_plus_punct)
            graph = delete_space + graph + delete_space
            graph |= punct

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logger.info(f"ClassifyFst grammars are saved to {far_file}.")
