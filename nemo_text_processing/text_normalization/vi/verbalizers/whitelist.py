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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)


class WhiteListFst(GraphFst):
    """
    Finite state transducer for verbalizing whitelist for Vietnamese
        e.g. tokens { name: "giờ" } -> giờ

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="whitelist", kind="verbalize", deterministic=deterministic)
        graph = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = graph @ pynini.cdrewrite(pynini.cross(u"\u00a0", NEMO_SPACE), "", "", NEMO_SIGMA)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
