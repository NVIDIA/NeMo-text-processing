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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. 십이 킬로그램 -> measure { cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
    """
    def __init__(self):
        super().__init__(name="measure", kind="verbalize")

        measurement = pynini.closure(NEMO_NOT_QUOTE, 1)

        optional_sign = pynini.closure(
            pynutil.delete('negative: "true"') 
            + delete_space 
            + pynutil.insert("-"), 
            0, 1
        )

        unit = (
            pynutil.delete('units: "')
            + measurement
            + pynutil.delete('"')
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_sign
            + delete_space
            + pynutil.delete('integer: "')
            + measurement
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + optional_sign
            + delete_space
            + pynutil.delete('integer_part: "')
            + measurement
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('fractional_part: "')
            + pynutil.insert(".")
            + measurement
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )

        graph_fraction = (
            pynutil.delete("fraction {")
            + delete_space
            + pynutil.delete('denominator: "')
            + measurement
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('numerator: "')
            + pynutil.insert("/")
            + measurement
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )

        graph = (
            (graph_cardinal | graph_decimal | graph_fraction)
            + delete_space 
            + pynutil.insert(" ")
            + unit
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()