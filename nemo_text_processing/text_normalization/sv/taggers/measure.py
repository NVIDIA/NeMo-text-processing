# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_space,
    delete_zero_or_one_space,
)
from nemo_text_processing.text_normalization.sv.graph_utils import SV_ALPHA, TO_LOWER
from nemo_text_processing.text_normalization.sv.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -12kg -> measure { negative: "true" cardinal { integer: "tolv" } units: "kilogram" }
        1kg -> measure { cardinal { integer: "ett" } units: "kilogram" }
        ,5kg -> measure { decimal { fractional_part: "fem" } units: "kilogram" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph_ett = cardinal.graph
        cardinal_graph_en = cardinal.graph_en

        graph_unit = pynini.string_file(get_abs_path("data/measure/unit.tsv"))
        graph_unit_ett = pynini.string_file(get_abs_path("data/measure/unit_neuter.tsv"))
        graph_plurals = pynini.string_file(get_abs_path("data/measure/unit_plural.tsv"))
        greek_lower = pynini.string_file(get_abs_path("data/measure/greek_lower.tsv"))
        greek_upper = pynutil.insert("stort ") + pynini.string_file(get_abs_path("data/measure/greek_upper.tsv"))
        greek = greek_lower | greek_upper

        graph_unit |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (SV_ALPHA | TO_LOWER) + pynini.closure(SV_ALPHA | TO_LOWER), graph_unit
        ).optimize()
        graph_unit_ett |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (SV_ALPHA | TO_LOWER) + pynini.closure(SV_ALPHA | TO_LOWER), graph_unit_ett
        ).optimize()

        graph_unit_plural = convert_space(graph_unit @ graph_plurals)
        graph_unit_plural_ett = convert_space(graph_unit_ett @ graph_plurals)
        graph_unit = convert_space(graph_unit)
        graph_unit_ett = convert_space(graph_unit_ett)
        self.unit_plural_en = graph_unit_plural
        self.unit_plural_ett = graph_unit_plural_ett
        self.unit_en = graph_unit
        self.unit_ett = graph_unit_ett

        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph_unit2 = (
            pynini.cross("/", "per")
            + delete_zero_or_one_space
            + pynutil.insert(NEMO_NON_BREAKING_SPACE)
            + (graph_unit | graph_unit_ett)
        )

        optional_graph_unit2 = pynini.closure(
            delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2, 0, 1,
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + pynutil.insert("\"")
        )
        unit_plural_ett = (
            pynutil.insert("units: \"")
            + (graph_unit_plural_ett + optional_graph_unit2 | graph_unit2)
            + pynutil.insert("\"")
        )

        unit_singular = (
            pynutil.insert("units: \"") + (graph_unit + optional_graph_unit2 | graph_unit2) + pynutil.insert("\"")
        )
        unit_singular_ett = (
            pynutil.insert("units: \"") + (graph_unit_ett + optional_graph_unit2 | graph_unit2) + pynutil.insert("\"")
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative_en
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )
        subgraph_decimal |= (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural_ett
        )

        # support radio FM/AM
        subgraph_decimal |= (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + pynutil.insert("units: \"")
            + pynini.union("AM", "FM")
            + pynutil.insert("\"")
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph_en)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph_ett)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural_ett
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("1", "ett")
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_singular_ett
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("1", "en")
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_singular
        )
        self.subgraph_cardinal = subgraph_cardinal

        unit_graph = (
            pynutil.insert("cardinal { integer: \"-\" } units: \"")
            + ((pynini.cross("/", "per") + delete_zero_or_one_space) | (pynini.accep("per") + pynutil.delete(" ")))
            + pynutil.insert(NEMO_NON_BREAKING_SPACE)
            + graph_unit
            + pynutil.insert("\" preserve_order: true")
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynini.cross('-', '')
            + pynutil.insert(" } units: \"")
            + pynini.closure(SV_ALPHA, 1)
            + pynutil.insert("\"")
        )

        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } units: \"")
            + (pynini.cross(pynini.union('x', "X"), 'x') | pynini.cross(pynini.union('x', "X"), ' times'))
            + pynutil.insert("\"")
        )

        alpha_dash_decimal = (
            pynutil.insert("units: \"")
            + pynini.closure(SV_ALPHA, 1)
            + pynini.accep('-')
            + pynutil.insert("\"")
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } preserve_order: true")
        )

        subgraph_fraction = (
            pynutil.insert("fraction { ") + fraction.graph + delete_space + pynutil.insert(" } ") + unit_plural
        )

        math_operations = pynini.string_file(get_abs_path("data/math_operations.tsv"))
        delimiter = pynini.accep(" ") | pynutil.insert(" ")

        equals = pynini.cross("=", "är")
        if not deterministic:
            equals |= pynini.cross("=", "är lika med")

        math = (
            (cardinal_graph_ett | SV_ALPHA | greek)
            + delimiter
            + math_operations
            + (delimiter | SV_ALPHA)
            + cardinal_graph_ett
            + delimiter
            + equals
            + delimiter
            + (cardinal_graph_ett | SV_ALPHA | greek)
        )

        math |= (
            (cardinal_graph_ett | SV_ALPHA | greek)
            + delimiter
            + equals
            + delimiter
            + (cardinal_graph_ett | SV_ALPHA)
            + delimiter
            + math_operations
            + delimiter
            + cardinal_graph_ett
        )

        math = (
            pynutil.insert("units: \"math\" cardinal { integer: \"")
            + math
            + pynutil.insert("\" } preserve_order: true")
        )
        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | unit_graph
            | decimal_dash_alpha
            | decimal_times
            | alpha_dash_decimal
            | subgraph_fraction
            | math
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_range(self, cardinal: GraphFst):
        """
        Returns range forms for measure tagger, e.g. 2-3, 2x3, 2*2

        Args:
            cardinal: cardinal GraphFst
        """
        range_graph = cardinal + pynini.cross(pynini.union("-", " - "), " till ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.cross(x, " gånger ") + cardinal

        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.cross(x, " gånger ") + cardinal
        return range_graph.optimize()
