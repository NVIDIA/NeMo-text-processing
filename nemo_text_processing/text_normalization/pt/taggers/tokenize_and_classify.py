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


import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.pt.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.pt.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.pt.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.pt.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.taggers.word import WordFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all Portuguese classification grammars. This class can process an entire sentence (lower cased).
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files.
        whitelist: path to a file with whitelist replacements.
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
                cache_dir,
                f"_{input_case}_pt_tn_{deterministic}_deterministic{whitelist_file}.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars. This might take some time...")

            # Initialize Portuguese taggers
            cardinal = CardinalFst(deterministic=deterministic)
            ordinal = OrdinalFst(cardinal, deterministic=deterministic)
            fraction = FractionFst(cardinal, ordinal, deterministic=deterministic)
            decimal = DecimalFst(cardinal, deterministic=deterministic)

            punctuation = PunctuationFst(deterministic=deterministic)
            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic).fst
            whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist)

            classify = (
                pynutil.add_weight(whitelist.fst, 1.01)
                | pynutil.add_weight(fraction.fst, 1.1)
                | pynutil.add_weight(decimal.fst, 1.1)
                | pynutil.add_weight(ordinal.fst, 1.1)
                | pynutil.add_weight(cardinal.fst, 1.1)
                | pynutil.add_weight(word_graph, 100)
            )

            # Wrap tokens properly
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            punct_graph = pynutil.insert("tokens { ") + pynutil.add_weight(punctuation.fst, weight=2.1) + pynutil.insert(" }")
            
            # Simple graph structure
            graph = token + pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space) + token
            )
            
            # Allow punctuation
            graph |= punct_graph

            self.fst = delete_space + graph + delete_space

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logger.info(f"ClassifyFst grammars are saved to {far_file}.")


if __name__ == "__main__":
    ClassifyFst(input_case="cased", deterministic=False)