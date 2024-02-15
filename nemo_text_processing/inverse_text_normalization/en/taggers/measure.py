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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    get_singulars,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. minus twelve kilograms -> measure { negative: "true" cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        # accept capital letters in units
        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_singular = pynini.invert(graph_unit)  # singular -> abbr
        graph_unit_singular = pynini.compose(casing_graph, graph_unit_singular).optimize()

        graph_unit_plural = get_singulars(graph_unit_singular).optimize()  # plural -> abbr
        graph_unit_plural = pynini.compose(casing_graph, graph_unit_plural).optimize()

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"true\"") + delete_extra_space, 0, 1,
        )

        unit_singular = convert_space(graph_unit_singular)
        unit_plural = convert_space(graph_unit_plural)
        unit_misc = pynutil.insert("/") + pynutil.delete("per") + delete_space + convert_space(graph_unit_singular)

        one_graph = pynini.accep("one").optimize()
        if input_case == INPUT_CASED:
            one_graph |= pynini.accep("One").optimize()

        unit_singular = (
            pynutil.insert("units: \"")
            + (unit_singular | unit_misc | pynutil.add_weight(unit_singular + delete_space + unit_misc, 0.01))
            + pynutil.insert("\"")
        )
        unit_plural = (
            pynutil.insert("units: \"")
            + (unit_plural | unit_misc | pynutil.add_weight(unit_plural + delete_space + unit_misc, 0.01))
            + pynutil.insert("\"")
        )

        # Let singular apply to values > 1 as they could be part of an adjective phrase (e.g. 14 foot tall building)
        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + (unit_plural | unit_singular)
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - one_graph) @ cardinal_graph)
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + (unit_plural | unit_singular)
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross(one_graph, "1")
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_singular
        )
        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
