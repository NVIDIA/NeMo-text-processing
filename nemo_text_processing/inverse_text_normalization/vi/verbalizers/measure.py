# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_NOT_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { negative: "true" cardinal { integer: "12" } units: "kg" } -> -12 kg

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst):
        super().__init__(name="measure", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-"), 0, 1)
        # Units that don't need space (time units)
        no_space_units = pynini.union("s", "ms", "ns", "μs", "h", "min", "%")

        unit_no_space = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete('"')
            + no_space_units
            + pynutil.delete('"')
            + delete_space
        )

        unit_with_space = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete('"')
            + (pynini.closure(NEMO_NOT_SPACE, 1) - no_space_units)
            + pynutil.delete('"')
            + delete_space
        )
        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + optional_sign
            + delete_space
            + decimal.numbers
            + delete_space
            + pynutil.delete("}")
        )
        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_sign
            + delete_space
            + cardinal.numbers
            + delete_space
            + pynutil.delete("}")
        )
        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        # Graph with no space for time units
        graph_no_space = (
            pynini.union(graph_cardinal, graph_decimal)
            + delete_space
            + optional_fractional
            + unit_no_space
            + delete_space
        )

        # Graph with space for other units
        graph_with_space = (
            pynini.union(graph_cardinal, graph_decimal)
            + delete_space
            + optional_fractional
            + insert_space
            + unit_with_space
            + delete_space
        )

        graph = pynini.union(graph_no_space, graph_with_space)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
