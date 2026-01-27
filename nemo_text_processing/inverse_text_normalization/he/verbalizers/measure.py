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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    delete_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, in Hebrew.
    Some measures are concatenated to the numbers and other are don't (two measure lists)
        e.g. measure { cardinal { integer: "3" } units: "מ״ג" } -> 3 מ״ג
        e.g. measure { cardinal { integer: "1000" } units: "%" } -> 1,000%
        e.g. measure { units: "%" cardinal { integer: "1" } } -> 1%
        e.g. measure { units: "ס״מ" cardinal { integer: "1" } } -> 1 ס״מ
        e.g. measure { prefix: "ל" cardinal { integer: "4" } units: "ס״מ" } -> ל-4 ס״מ

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst):
        super().__init__(name="measure", kind="verbalize")

        optional_prefix = pynini.closure(
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.insert("-")
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )

        # Removes the negative attribute and leaves the sign if occurs
        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete('"')
            + pynini.accep("-")
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + optional_sign
            + decimal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_sign
            + cardinal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        unit = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_CHAR - NEMO_SPACE, 1)
            + pynutil.delete('"')
            + delete_space
        )
        unit @= pynini.cdrewrite(
            pynini.cross("\[SPACE\]", NEMO_SPACE), "", "", NEMO_SIGMA  # noqa: W605
        )  # For space separated measures.

        numbers_units = delete_space + unit
        numbers_graph = (graph_cardinal | graph_decimal) + numbers_units

        one_graph = delete_space + pynutil.insert("1") + unit + pynutil.delete('cardinal { integer: "1" }')

        graph = optional_prefix + (numbers_graph | one_graph)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
