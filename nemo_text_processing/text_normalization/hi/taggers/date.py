# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_HI_DIGIT,
    NEMO_HI_NON_ZERO,
    NEMO_HI_ZERO,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

days = pynini.string_file(get_abs_path("data/date/days.tsv"))
months = pynini.string_file(get_abs_path("data/date/months.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "०१-०४-२०२४" -> date { day: "एक" month: "अप्रैल" year: "दो हज़ार चौबीस" }
        "०४-०१-२०२४" -> date { month: "अप्रैल" day: "एक" year: "दो हज़ार चौबीस" }
        

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        graph_year_thousands = pynini.compose(
            (NEMO_HI_DIGIT + NEMO_HI_ZERO + NEMO_HI_DIGIT + NEMO_HI_DIGIT), cardinal.graph_thousands
        )
        graph_year_hundreds_as_thousands = pynini.compose(
            (NEMO_HI_DIGIT + NEMO_HI_NON_ZERO + NEMO_HI_DIGIT + NEMO_HI_DIGIT), cardinal.graph_hundreds_as_thousand
        )

        graph_year = graph_year_thousands | graph_year_hundreds_as_thousands

        delete_dash = pynutil.delete("-")
        delete_slash = pynutil.delete("/")

        days_graph = pynutil.insert("day: \"") + days + pynutil.insert("\"") + insert_space

        months_graph = pynutil.insert("month: \"") + months + pynutil.insert("\"") + insert_space

        years_graph = pynutil.insert("year: \"") + graph_year + pynutil.insert("\"") + insert_space

        graph_dd_mm = days_graph + delete_dash + months_graph

        graph_mm_dd = months_graph + delete_dash + days_graph

        graph_mm_dd += pynutil.insert(" preserve_order: true ")

        graph_dd_mm_yyyy = (
            days_graph + (delete_dash | delete_slash) + months_graph + (delete_dash | delete_slash) + years_graph
        )

        graph_mm_dd_yyyy = (
            months_graph + (delete_dash | delete_slash) + days_graph + (delete_dash | delete_slash) + years_graph
        )

        graph_mm_dd_yyyy += pynutil.insert(" preserve_order: true ")

        graph_mm_yyyy = months_graph + delete_dash + years_graph

        # default assume dd_mm_yyyy

        final_graph = (
            pynutil.add_weight(graph_dd_mm, -0.001)
            | graph_mm_dd
            | pynutil.add_weight(graph_dd_mm_yyyy, -0.001)
            | graph_mm_dd_yyyy
            | graph_mm_yyyy
        )

        self.final_graph = final_graph.optimize()

        self.fst = self.add_tokens(self.final_graph)
