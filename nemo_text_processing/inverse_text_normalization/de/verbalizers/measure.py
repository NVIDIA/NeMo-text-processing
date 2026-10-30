# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measures:
        e.g. measure { cardinal { integer: "2" } units: "kg" } -> 2 kg
        e.g. measure { cardinal { negative: "-" integer: "67" } units: "kg" } -> -67 kg
        e.g. measure { decimal { integer_part: "1" fractional_part: "5" } units: "km/h" } -> 1,5 km/h
        e.g. measure { fraction { numerator: "3" denominator: "4" } units: "ω" } -> 3/4 ω
        e.g. measure { decimal { integer_part: "1" fractional_part: "50" } units: "m" } -> 1,50 m
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst):
        super().__init__(name="measure", kind="verbalize")
        graph_cardinals = cardinal.numbers
        graph_decimals = decimal.numbers
        graph_fractions = fraction.numbers
        graph_negative = pynini.cross('negative: "-"', "-") + pynutil.delete(NEMO_SPACE)

        graph_unit = (
            pynutil.delete("units:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + (pynini.closure(NEMO_NOT_QUOTE, 1) | pynini.closure(pynini.difference(NEMO_CHAR, NEMO_SPACE), 1))
            + pynutil.delete('"')
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_cardinals
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_decimals
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        graph_fraction = (
            pynutil.delete("fraction {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_fractions
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        graph = (graph_decimal | graph_cardinal | graph_fraction) + pynini.accep(NEMO_SPACE) + graph_unit

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
