# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_SIGMA, GraphFst, generator_main
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
            far_file = os.path.join(cache_dir, "ko_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.fst = self.get_postprocess_graph()
            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_postprocess_graph(self):
        """
        Build and return the post-processing FST.
        """
        sigma = pynini.closure(NEMO_SIGMA)

        # Collapse spaces around the particle "부터"
        delete_space_around_particle = pynini.cdrewrite(
            pynini.cross(" 부터 ", "부터"),
            "",
            "",
            sigma,
        )

        # Join "<Month> <day-word> ... 부터" -> "<Month><day-word>부터"
        SPACE = pynini.accep(" ")
        BUHTEO = pynini.accep("부터")

        # Month words in Korean TN output
        MONTH_WORD = pynini.union(
            "일월", "이월", "삼월", "사월", "오월", "유월",
            "칠월", "팔월", "구월", "시월", "십일월", "십이월",
        )

        # First syllable of the day number word (enough to detect the pattern)
        NUMHEAD = pynini.union("일", "이", "삼", "사", "오", "육", "칠", "팔", "구", "십")

        rm_space_month_num_bu = pynini.cdrewrite(
            pynini.cross(" ", ""),                     
            MONTH_WORD,                                 
            NUMHEAD + pynini.closure(SPACE) + BUHTEO,
            sigma,
        )

        # Apply Rule 1, then Rule 2
        return (delete_space_around_particle @ rm_space_month_num_bu).optimize()







