# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
    generate_far_filename,
)
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.word import WordFst
from nemo_text_processing.utils.logging import logger


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "अब" } tokens { time { hours: "१२" minutes: "३०" } } tokens { name: "बज" } tokens { name: "गए" } tokens { name: "हैं" } -> अब १२:३० बज गए हैं
    """

    def __init__(
        self,
        project_input: bool = False,
        cache_dir: str = None,
        overwrite_cache: bool = False
    ):
        super().__init__(name="verbalize_final", kind="verbalize", project_input=project_input)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = generate_far_filename(
                language="hi",
                mode="itn",
                cache_dir=cache_dir,
                operation="verbalize",
                project_input=project_input
            )
        
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
            logger.info(f'VerbalizeFinalFst graph was restored from {far_file}.')
        else:
            verbalize = VerbalizeFst(project_input=project_input).fst
            word = WordFst(project_input=project_input).fst
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
            graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space
            
            self.fst = graph.optimize()
            if far_file:
                generator_main(far_file, {"verbalize": self.fst})
                logger.info(f"VerbalizeFinalFst grammars are saved to {far_file}.")
