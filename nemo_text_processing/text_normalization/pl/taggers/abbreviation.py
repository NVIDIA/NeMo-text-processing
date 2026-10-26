# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.pl.graph_utils import PL_UPPER


class AbbreviationFst(GraphFst):
    """Classifies uppercase Polish initialisms, with or without dots."""

    def __init__(self, whitelist=None, deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")
        graph = PL_UPPER + dot + pynini.closure(insert_space + PL_UPPER + dot, 1)
        graph |= PL_UPPER + dot + pynini.closure(PL_UPPER + dot, 1)
        graph |= PL_UPPER + pynini.closure(insert_space + PL_UPPER, 1)

        if whitelist is not None:
            graph = pynini.compose(
                pynini.difference(pynini.project(graph, "input"), pynini.project(whitelist.graph, "input")), graph
            )

        graph = pynutil.insert('value: "') + graph.optimize() + pynutil.insert('"')
        self.fst = self.add_tokens(graph).optimize()
