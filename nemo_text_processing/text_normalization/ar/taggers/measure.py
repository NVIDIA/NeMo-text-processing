# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path

unit_singular = pynini.string_file(get_abs_path("data/measure/measurements.tsv"))


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure,  e.g.
        "20%" -> measure { cardinal { integer: "20" } units: "%" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.cardinal_numbers
        graph_unit = pynini.string_file(get_abs_path("data/measure/measurements.tsv"))

        graph_unit_singular = convert_space(graph_unit)
        optional_graph_negative = pynini.closure("-", 0, 1)

        graph_unit_denominator = pynini.cross("/", "في") + pynutil.insert(
            NEMO_NON_BREAKING_SPACE
        ) + graph_unit_singular | graph_unit_singular + pynutil.insert(NEMO_NON_BREAKING_SPACE) + pynini.cross(
            "/", "في"
        )

        optional_unit_denominator = pynini.closure(
            pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit_denominator, 0, 1,
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (graph_unit_singular + (optional_unit_denominator) | graph_unit_denominator)
            + pynutil.insert("\"")
        )

        unit_singular_graph = (
            pynutil.insert("units: \"")
            + ((graph_unit_singular + optional_unit_denominator) | graph_unit_denominator)
            + pynutil.insert("\"")
        )

        subgraph_decimal = (
            decimal.fst + insert_space + pynini.closure(pynutil.delete(" "), 0, 1) + unit_plural
            | unit_plural + pynini.closure(pynutil.delete(" "), 0, 1) + insert_space + decimal.fst
        )

        subgraph_cardinal = (
            (optional_graph_negative + (pynini.closure(NEMO_DIGIT) - "1")) @ cardinal.fst
            + insert_space
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + unit_plural
            | unit_plural
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + insert_space
            + (optional_graph_negative + (pynini.closure(NEMO_DIGIT) - "1")) @ cardinal.fst
        )

        subgraph_cardinal |= (
            (optional_graph_negative + pynini.accep("1")) @ cardinal.fst
            # @ pynini.cdrewrite(pynini.cross("واحد", ""), "", "", NEMO_SIGMA)
            + insert_space
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + unit_singular_graph
        )

        subgraph_fraction = fraction.fst + insert_space + pynini.closure(pynutil.delete(" "), 0, 1) + unit_plural
        subgraph_fraction |= unit_plural + pynini.closure(pynutil.delete(" "), 0, 1) + insert_space + fraction.fst

        cardinal_dash_alpha = (
            pynutil.insert("cardinal { integer: \"")
            + cardinal_graph
            + pynutil.delete('-')
            + pynutil.insert("\" } units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_cardinal = (
            pynutil.insert("units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.delete('-')
            + pynutil.insert("\"")
            + pynutil.insert(" cardinal { integer: \"")
            + cardinal_graph
            + pynutil.insert("\" }")
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_decimal
            + pynutil.delete('-')
            + pynutil.insert(" } units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert("\"")
        )

        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_decimal
            + pynutil.insert(" } units: \"")
            + pynini.union('x', 'X')
            + pynutil.insert("\"")
        )

        cardinal_times = (
            pynutil.insert("cardinal { integer: \"")
            + cardinal_graph
            + pynutil.insert("\" } units: \"")
            + pynini.union('x', 'X')
            + pynutil.insert("\"")
        )

        alpha_dash_decimal = (
            pynutil.insert("units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.delete('-')
            + pynutil.insert("\"")
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_decimal
            + pynutil.insert(" }")
        )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | cardinal_dash_alpha
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | decimal_times
            | alpha_dash_decimal
            | subgraph_fraction
            | cardinal_times
        )
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
