# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure (pt-BR), e.g.
        measure { cardinal { integer: "duzentos" } units: "gramas" } -> duzentos gramas
        measure { cardinal { integer: "um" } units: "hora" } -> uma hora

    Args:
        decimal: DecimalFst verbalizer
        cardinal: CardinalFst verbalizer
        fraction: FractionFst verbalizer
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        hours_unit = (
            pynutil.delete('units: "')
            + pynini.union(pynini.accep("hora"), pynini.accep("horas"))
            + pynutil.delete('"')
        )
        non_hours_unit = (
            pynutil.delete('units: "')
            + pynini.difference(pynini.closure(NEMO_NOT_QUOTE, 1), pynini.union("hora", "horas"))
            + pynutil.delete('"')
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + cardinal.graph_fem
            + delete_space
            + pynutil.delete("}")
            + NEMO_WHITE_SPACE
            + hours_unit
        )
        graph_cardinal |= (
            pynutil.delete("cardinal {")
            + delete_space
            + cardinal.graph_masc
            + delete_space
            + pynutil.delete("}")
            + NEMO_WHITE_SPACE
            + non_hours_unit
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + decimal.numbers
            + delete_space
            + pynutil.delete("}")
            + NEMO_WHITE_SPACE
            + (hours_unit | non_hours_unit)
        )

        graph_fraction = (
            pynutil.delete("fraction {")
            + delete_space
            + fraction.inner_graph
            + delete_space
            + pynutil.delete("}")
            + NEMO_WHITE_SPACE
            + (hours_unit | non_hours_unit)
        )

        graph = graph_cardinal | graph_decimal | graph_fraction
        graph += delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
