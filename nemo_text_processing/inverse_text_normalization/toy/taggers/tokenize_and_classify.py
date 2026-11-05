# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.bantu.taggers.tokenize_and_classify import build_classifier
from nemo_text_processing.inverse_text_normalization.toy.settings import load
from nemo_text_processing.text_normalization.en.graph_utils import INPUT_LOWER_CASED, GraphFst, generator_main
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars for the test language
    of the shared Bantu kernel. The grammar is built by the kernel from this language's
    data/ and settings.py; see bantu.taggers.tokenize_and_classify.build_classifier.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        whitelist: not used; the test language has no whitelist
        overwrite_cache: set to True to overwrite .far files
        input_case: accepting either "lower_cased" or "cased" input. The kernel reads
            lower-cased words and passes other words through, so both build the same grammar.
    """

    def __init__(
        self,
        cache_dir: str = None,
        whitelist: str = None,
        overwrite_cache: bool = False,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"toy_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info("Creating ClassifyFst grammars.")
            self.fst = build_classifier(load())
            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logger.info(f"ClassifyFst grammars are saved to {far_file}.")
