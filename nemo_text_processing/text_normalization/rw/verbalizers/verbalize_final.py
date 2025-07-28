# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.rw.graph_utils import GraphFst, delete_space, generator_main
from nemo_text_processing.text_normalization.rw.verbalizers.verbalize import VerbalizeFst


class VerbalizeFinalFst(GraphFst):
    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False, deterministic: bool = True):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"rw_tn_verbalizer.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
        else:
            verbalize = VerbalizeFst(deterministic=deterministic).fst
            word = WordFst(deterministic=deterministic).fst
            types = verbalize | word
            graph = (
                pynutil.delete("tokens")
                + delete_space
                + pynutil.delete("{")
                + delete_space
                + types
                + delete_space
                + pynutil.delete("}")
            )
            graph = delete_space + pynini.closure(graph + delete_space) + graph + delete_space

            self.fst = graph

            if far_file:
                generator_main(far_file, {"ALL": self.fst, 'REDUP': pynini.accep("REDUP")})
