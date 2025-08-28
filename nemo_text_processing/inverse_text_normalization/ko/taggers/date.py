# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import GraphFst, NEMO_SPACE
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. 이천십이년 일월 오일 -> date { year: "2012" month: "1" day: "5"  }
        e.g. 오월 -> date { month: "5" }
        e.g. 칠일 -> date { day: "7" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        cardinal = cardinal.just_cardinals
        month = pynini.string_file(get_abs_path("data/months.tsv"))

        spacing = pynini.closure(pynini.accep(NEMO_SPACE), 0, 1)

        year_suffix = pynini.cross("년", "")
        month_suffix = pynini.cross("월", "")
        day_suffix = pynini.cross("일", "")

        year_component = (
            pynutil.insert("year: \"")
            + cardinal
            + pynini.closure(year_suffix, 0, 1)
            + pynutil.insert("\"")
        )

        month_component = (
            pynutil.insert("month: \"")
            + spacing
            + month
            + pynini.closure(month_suffix, 0, 1)
            + pynutil.insert("\"")
        )

        day_component = (
            pynutil.insert("day: \"")
            + spacing
            + cardinal
            + day_suffix
            + spacing
            + pynutil.insert("\"")
        )

        graph_component = year_component | month_component | day_component
        graph_date = (
            pynini.closure(year_component, 0, 1)
            + pynini.closure((pynutil.insert(NEMO_SPACE)) + month_component, 0, 1)
            + pynini.closure((pynutil.insert(NEMO_SPACE)) + day_component, 0, 1)
        )

        final_graph = graph_component | graph_date

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

