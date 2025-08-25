# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst, delete_space, generator_main
from nemo_text_processing.text_normalization.ko.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.utils.logging import logger


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "its" } tokens { time { hours: "twelve" minutes: "thirty" } } tokens { name: "now" } tokens { name: "." } -> its twelve thirty now .

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """
    
    def __init__(self, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"ko_tn_{deterministic}_verbalizer.far")

        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
            logger.info(f'VerbalizeFinalFst graph was restored from {far_file}.')
        else:
            token_graph = VerbalizeFst(deterministic=deterministic)

            token_verbalizer = (
                pynutil.delete("tokens {") + delete_space + token_graph.fst + delete_space + pynutil.delete(" }")
            )

            verbalizer = pynini.closure(delete_space + token_verbalizer + delete_space)

            self.fst = verbalizer.optimize()

            if far_file:
                generator_main(far_file, {"verbalize": self.fst})
