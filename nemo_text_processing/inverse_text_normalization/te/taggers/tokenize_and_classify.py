# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.inverse_text_normalization.te.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.te.taggers.word import WordFst


class ClassifyFst(GraphFst):
    """
    Final classification grammar for Telugu ITN cardinal processing.
    """

    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        input_case: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "te_itn.far")

        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info("Creating Telugu ClassifyFst grammar.")

            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst
            word_graph = WordFst().fst

            classify = pynutil.add_weight(cardinal_graph, 1.1) | pynutil.add_weight(word_graph, 100)

            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")

            graph = token + pynini.closure(pynutil.add_weight(delete_extra_space + token, 1000.0))
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(
                    far_file,
                    {"tokenize_and_classify": self.fst},
                )
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
