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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.hi.graph_utils import MINUS, NEMO_NOT_QUOTE, GraphFst, insert_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. fraction { integer: "तेईस" numerator: "चार" denominator: "छः" }-> तेईस चार बटा छः
        e.g. fraction { numerator: "चार" denominator: "छः" } -> चार बटा छः
        

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", MINUS) + pynutil.delete(" "), 0, 1)

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        denominator = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        insert_bata = pynutil.insert(" बटा ")

        fraction_default = numerator + insert_bata + denominator

        self.graph = optional_sign + pynini.closure(pynini.closure(integer, 0, 1) + insert_space) + fraction_default

        graph = self.graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
