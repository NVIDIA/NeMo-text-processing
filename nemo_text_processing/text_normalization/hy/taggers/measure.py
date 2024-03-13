# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
)
from nemo_text_processing.text_normalization.hy.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. 52 կգ -> measure { cardinal { integer: "հիսուներկու" } units: "կիլոգրամ" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.all_nums_no_tokens

        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA)

        graph_measurements_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_measurements_unit = pynini.compose(casing_graph, graph_measurements_unit)

        graph_measurements_dates_unit = pynini.string_file(get_abs_path("data/measurement_dates.tsv"))
        graph_measurements_dates_unit = pynini.compose(casing_graph, graph_measurements_dates_unit)

        measurements_unit = pynutil.insert("units: \"") + graph_measurements_unit + pynutil.insert("\"")

        measurements_dates_unit = pynutil.insert("units: \"") + graph_measurements_dates_unit + pynutil.insert("\"")

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space, 0, 1)
            + measurements_unit
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space, 0, 1)
            + measurements_unit
        )

        subgraph_cardinal_dates = (
            pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space, 0, 1)
            + measurements_dates_unit
        )

        subgraph_cardinal_dates |= (
            pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("ից")
            + pynutil.delete("-")
            + pynutil.insert(' ')
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space, 0, 1)
            + measurements_dates_unit
        )

        final_graph = subgraph_decimal | subgraph_cardinal | subgraph_cardinal_dates
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
