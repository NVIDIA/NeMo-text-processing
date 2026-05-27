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

from nemo_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_CHAR,
    NEMO_SIGMA,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.utils.logging import logger


class PostProcessingFst:
    """
    Post-processes a fully verbalized Spanish sentence, e.g. removes spurious spaces
    before punctuation: ``palabra , otra`` -> ``palabra, otra``.
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "es_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f"Post processing graph was restored from {far_file}.")
        else:
            self.fst = self.get_punct_postprocess_graph()
            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_punct_postprocess_graph(self):
        """
        Removes spaces before punctuation marks (comma, period, etc.) while keeping
        spaces before quotes, dashes, and opening brackets.
        """
        punct_marks_all = PunctuationFst().punct_marks

        quotes = ["'", '"', "«", "¿", "¡"]
        dashes = ["-", "—"]
        brackets = ["<", "{", "(", r"\["]
        allow_space_before_punct = quotes + dashes + brackets

        no_space_before_punct = [
            m for m in punct_marks_all if m not in allow_space_before_punct and m != "."
        ]
        # Keep a space before "." (e.g. "punto net ." for "www.enveedya.net."); only strip before , ; : ! ?
        no_space_before_punct = list(set(no_space_before_punct + [",", ";", ":", "!", "?"]))
        no_space_before_punct = pynini.union(*no_space_before_punct)

        delete_space = pynutil.delete(" ")
        non_punct = pynini.difference(NEMO_CHAR, no_space_before_punct).optimize()
        graph = (
            pynini.closure(non_punct)
            + pynini.closure(
                no_space_before_punct | pynutil.add_weight(delete_space + no_space_before_punct, MIN_NEG_WEIGHT)
            )
            + pynini.closure(non_punct)
        )
        graph = pynini.closure(graph).optimize()

        no_space_after_punct = pynini.union(*brackets)
        no_space_after_punct = pynini.cdrewrite(delete_space, no_space_after_punct, NEMO_SIGMA, NEMO_SIGMA).optimize()
        return pynini.compose(graph, no_space_after_punct).optimize()
