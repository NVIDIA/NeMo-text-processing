# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_CHAR,
    NEMO_SIGMA,
    generator_main,
)
from nemo_text_processing.text_normalization.hi.taggers.punctuation import PunctuationFst
from nemo_text_processing.utils.logging import logger


class PostProcessingFst:
    """
    Finite state transducer that post-processing an entire sentence after verbalization is complete, e.g.
    removes extra spaces around punctuation marks " ( one hundred and twenty three ) " -> "(one hundred and twenty three)"

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "hi_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.fst = self.get_punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_punct_postprocess_graph(self):
        """
        Returns graph to post process punctuation marks.

        By default, spaces are removed before punctuation marks like comma, period, etc.
        """
        punct_marks_all = PunctuationFst().punct_marks

        # Punctuation marks that should NOT have space before them
        # (most punctuation except quotes, dashes, and opening brackets)
        quotes = ["'", "\"", "«"]
        dashes = ["-", "—"]
        brackets = ["<", "{", "(", r"\["]
        allow_space_before_punct = quotes + dashes + brackets

        no_space_before_punct = [m for m in punct_marks_all if m not in allow_space_before_punct]
        # Add Hindi-specific punctuation
        no_space_before_punct.extend(["।", ",", ".", ";", ":", "!", "?"])
        # Remove duplicates
        no_space_before_punct = list(set(no_space_before_punct))
        no_space_before_punct = pynini.union(*no_space_before_punct)

        delete_space = pynutil.delete(" ")

        # Delete space before no_space_before_punct marks
        non_punct = pynini.difference(NEMO_CHAR, no_space_before_punct).optimize()
        graph = (
            pynini.closure(non_punct)
            + pynini.closure(
                no_space_before_punct | pynutil.add_weight(delete_space + no_space_before_punct, MIN_NEG_WEIGHT)
            )
            + pynini.closure(non_punct)
        )
        graph = pynini.closure(graph).optimize()

        # Remove space after opening brackets
        no_space_after_punct = pynini.union(*brackets)
        no_space_after_punct = pynini.cdrewrite(delete_space, no_space_after_punct, NEMO_SIGMA, NEMO_SIGMA).optimize()
        graph = pynini.compose(graph, no_space_after_punct).optimize()

        return graph
