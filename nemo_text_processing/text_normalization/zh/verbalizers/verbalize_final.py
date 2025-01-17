# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst, delete_space, generator_main
from nemo_text_processing.text_normalization.zh.verbalizers.postprocessor import PostProcessor
from nemo_text_processing.text_normalization.zh.verbalizers.verbalize import VerbalizeFst
from pynini.lib import pynutil

# import logging


class VerbalizeFinalFst(GraphFst):
    """

    """

    def __init__(self, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"zh_tn_{deterministic}_deterministic_verbalizer.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
        else:
            token_graph = VerbalizeFst(deterministic=deterministic)
            token_verbalizer = (
                pynutil.delete("tokens {") + delete_space + token_graph.fst + delete_space + pynutil.delete(" }")
            )
            verbalizer = pynini.closure(delete_space + token_verbalizer + delete_space)

            postprocessor = PostProcessor(remove_puncts=False, to_upper=False, to_lower=False, tag_oov=False,)

            self.fst = (verbalizer @ postprocessor.fst).optimize()
            if far_file:
                generator_main(far_file, {"verbalize": self.fst})
