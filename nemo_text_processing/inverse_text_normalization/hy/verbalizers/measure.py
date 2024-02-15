# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SPACE, GraphFst, delete_space


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "59" } units: "Հց" } -> 59 Հց

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst):
        super().__init__(name="measure", kind="verbalize")
        unit = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - NEMO_SPACE, 1)
            + pynutil.delete("\"")
            + delete_space
        )
        graph_decimal = (
            pynutil.delete("decimal {") + delete_space + decimal.numbers + delete_space + pynutil.delete("}")
        )
        graph_cardinal_first = (
            pynutil.delete("cardinal {") + delete_space + cardinal.numbers + delete_space + pynutil.delete("} ")
        )

        graph_cardinal_two = (
            pynutil.delete("cardinal {")
            + pynutil.delete(" integer: \"")
            + delete_space
            + pynini.closure(NEMO_CHAR - NEMO_SPACE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("} ")
        )

        graph_first = (graph_cardinal_first | graph_decimal) + delete_space + pynutil.insert(" ") + unit
        graph_second = graph_cardinal_two + delete_space + pynutil.insert(" ") + unit
        graph = graph_first | graph_second
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
