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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese ordinals, e.g.
        ordinal { integer: "nhất" } -> thứ nhất
        ordinal { integer: "tư" } -> thứ tư
        ordinal { integer: "mười lăm" } -> thứ mười lăm
        ordinal { integer: "một trăm" } -> thứ một trăm

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)

        quoted_content = pynini.closure(NEMO_NOT_QUOTE)

        integer = (
            pynutil.delete("integer:") + delete_space + pynutil.delete("\"") + quoted_content + pynutil.delete("\"")
        )

        ordinal_pattern = pynutil.insert("thứ ") + integer

        self.ordinal_graph = ordinal_pattern

        delete_tokens = self.delete_tokens(self.ordinal_graph)
        self.fst = delete_tokens.optimize()
