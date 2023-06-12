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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_space,
    delete_zero_or_one_space,
)
from nemo_text_processing.text_normalization.se.graph_utils import SE_ALPHA, TO_LOWER
from nemo_text_processing.text_normalization.se.utils import get_abs_path
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -12kg -> measure { negative: "true" cardinal { integer: "guoktenuppelohkái" } units: "kilográmma" }
        1kg -> measure { cardinal { integer: "okta" } units: "kilográmma" }
        ,5kg -> measure { decimal { fractional_part: "guhtta" } units: "kilográmma" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        unit_simple = pynini.string_file(get_abs_path("data/measure/unit_simple.tsv"))
        simple_endings = pynini.string_file(get_abs_path("data/inflection/simple.tsv"))
        graph_simple = unit_simple | unit_simple + simple_endings

        graph_plurals = pynini.string_file(get_abs_path("data/measure/unit_plural.tsv"))
        greek_lower = pynini.string_file(get_abs_path("data/measure/greek_lower.tsv"))
        greek_upper = pynutil.insert("stort ") + pynini.string_file(get_abs_path("data/measure/greek_lower.tsv"))
        greek = greek_lower | greek_upper

        graph_unit |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (SE_ALPHA | TO_LOWER) + pynini.closure(SE_ALPHA | TO_LOWER), graph_unit
        ).optimize()

        graph_unit_plural = convert_space(graph_unit @ graph_plurals)
        graph_unit = convert_space(graph_unit)

        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph_unit2 = (
            pynini.cross("/", "per") + delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit
        )

        optional_graph_unit2 = pynini.closure(
            delete_zero_or_one_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2, 0, 1,
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + pynutil.insert("\"")
        )

        unit_singular = (
            pynutil.insert("units: \"") + (graph_unit + optional_graph_unit2 | graph_unit2) + pynutil.insert("\"")
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative_en
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
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
            + (NEMO_SIGMA @ cardinal_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
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
            + pynini.closure(SE_ALPHA, 1)
            + pynutil.insert("\"")
        )

        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } units: \"")
            + (pynini.cross(pynini.union('x', "X"), 'x') | pynini.cross(pynini.union('x', "X"), ' geardde'))
            + pynutil.insert("\"")
        )

        alpha_dash_decimal = (
            pynutil.insert("units: \"")
            + pynini.closure(SE_ALPHA, 1)
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

        equals = pynini.cross("=", "lea")

        math = (
            (cardinal_graph | SE_ALPHA | greek)
            + delimiter
            + math_operations
            + (delimiter | SE_ALPHA)
            + cardinal_graph
            + delimiter
            + equals
            + delimiter
            + (cardinal_graph | SE_ALPHA | greek)
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
