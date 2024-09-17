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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    delete_space,
    generator_main,
)
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
            far_file = os.path.join(cache_dir, "zh_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.set_punct_dict()
            self.fst = self.get_punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def set_punct_dict(self):
        self.punct_marks = {
            "'": [
                "'",
                '´',
                'ʹ',
                'ʻ',
                'ʼ',
                'ʽ',
                'ʾ',
                'ˈ',
                'ˊ',
                'ˋ',
                '˴',
                'ʹ',
                '΄',
                '՚',
                '՝',
                'י',
                '׳',
                'ߴ',
                'ߵ',
                'ᑊ',
                'ᛌ',
                '᾽',
                '᾿',
                '`',
                '´',
                '῾',
                '‘',
                '’',
                '‛',
                '′',
                '‵',
                'ꞌ',
                '＇',
                '｀',
                '𖽑',
                '𖽒',
            ],
        }

    def get_punct_postprocess_graph(self):
        """
        Returns graph to post process punctuation marks.

        {``} quotes are converted to {"}. Note, if there are spaces around single quote {'}, they will be kept.
        By default, a space is added after a punctuation mark, and spaces are removed before punctuation marks.
        """

        remove_space_around_single_quote = pynini.cdrewrite(
            delete_space, NEMO_NOT_SPACE, NEMO_NOT_SPACE, pynini.closure(NEMO_SIGMA)
        )
        # this works if spaces in between (good)
        # delete space between 2 NEMO_NOT_SPACE（left and right to the space) that are with in a content of NEMO_SIGMA

        graph = remove_space_around_single_quote.optimize()

        return graph
