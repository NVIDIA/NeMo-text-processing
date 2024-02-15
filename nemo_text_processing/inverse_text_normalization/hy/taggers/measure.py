# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.hy.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_extra_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. իննսունյոթ հերց -> measure { cardinal { integer: "97" } units: "Հց" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        input_case: accepting either "lower_cased" or "cased" input.
        (input_case is not necessary everything is made for lower_cased input)
        TODO add cased input support
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        from_to = pynini.string_map([("ից", "")])
        cardinal_graph += pynutil.insert("") | from_to

        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()

        graph_measurements_unit = pynini.string_file(get_abs_path("data/measurements.tsv")) + (
            pynutil.insert("") | pynutil.insert("ում") | pynutil.insert("ից")
        )
        graph_measurements_unit = pynini.invert(graph_measurements_unit)
        graph_measurements_unit = pynini.compose(casing_graph, graph_measurements_unit).optimize()

        measurements_unit = convert_space(graph_measurements_unit)

        graph_measurements_dates_unit = pynini.string_file(get_abs_path("data/measurement_dates.tsv"))

        graph_measurements_dates_unit = pynini.invert(graph_measurements_dates_unit)
        graph_measurements_dates_unit = pynini.compose(casing_graph, graph_measurements_dates_unit).optimize()

        measurements_dates_unit = convert_space(graph_measurements_dates_unit)

        measurements_unit = pynutil.insert("units: \"") + measurements_unit + pynutil.insert("\"")

        measurements_dates_unit = pynutil.insert("units: \"") + measurements_dates_unit + pynutil.insert("\"")

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + measurements_unit
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + measurements_unit
        )
        subgraph_cardinal_dates = (
            (measurements_dates_unit + delete_extra_space | pynutil.insert(""))
            + pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + measurements_dates_unit
        )
        subgraph_cardinal_dates |= (
            (measurements_dates_unit + delete_extra_space | pynutil.insert(""))
            + pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert('-')
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + measurements_dates_unit
        )

        final_graph = subgraph_decimal | subgraph_cardinal | subgraph_cardinal_dates
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
