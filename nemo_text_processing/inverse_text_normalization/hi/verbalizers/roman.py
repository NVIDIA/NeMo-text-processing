# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing Roman numerals
        e.g. tokens { roman { key_cardinal: "अध्याय" integer: "III" } } -> अध्याय III
    """

    def __init__(self):
        super().__init__(name="roman", kind="verbalize")
        key_cardinal = pynutil.delete("key_cardinal: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph = key_cardinal + delete_space + insert_space + integer
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
