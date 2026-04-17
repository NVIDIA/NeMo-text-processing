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
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure (pt-BR), e.g.
        200 g -> measure { cardinal { integer: "duzentos" } units: "gramas" }
        1 kg -> measure { cardinal { integer: "um" } units: "quilo" }
        2,4 g -> measure { decimal { ... } units: "gramas" }
        1/2 l -> measure { fraction { ... } units: "litros" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        unit_singular = pynini.string_file(get_abs_path("data/measure/measurements_singular.tsv"))
        unit_plural = pynini.string_file(get_abs_path("data/measure/measurements_plural.tsv"))

        graph_unit_singular = convert_space(unit_singular)
        graph_unit_plural = convert_space(unit_plural)

        optional_graph_negative = pynini.closure(pynini.accep("-"), 0, 1)

        unit_plural = pynutil.insert('units: "') + graph_unit_plural + pynutil.insert('"')
        unit_singular_graph = pynutil.insert('units: "') + graph_unit_singular + pynutil.insert('"')

        subgraph_decimal = decimal.fst + insert_space + pynini.closure(NEMO_SPACE, 0, 1) + unit_plural

        subgraph_cardinal = (
            (optional_graph_negative + (NEMO_SIGMA - "1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + unit_plural
        )

        subgraph_cardinal |= (
            (optional_graph_negative + pynini.accep("1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + unit_singular_graph
        )

        subgraph_fraction = fraction.fst + insert_space + pynini.closure(delete_space, 0, 1) + unit_plural

        final_graph = subgraph_decimal | subgraph_cardinal | subgraph_fraction
        self.fst = self.add_tokens(final_graph).optimize()
