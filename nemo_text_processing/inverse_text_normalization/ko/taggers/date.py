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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_SPACE, GraphFst
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

        year_suffix = pynini.cross("년", "")
        month_suffix = pynini.cross("월", "")
        day_suffix = pynini.cross("일", "")

        delete_space = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1)
        between_fields = delete_space + pynutil.insert(NEMO_SPACE)

        year_component = pynutil.insert("year: \"") + cardinal + year_suffix + pynutil.insert("\"")
        month_component = pynutil.insert("month: \"") + month + month_suffix + pynutil.insert("\"")
        day_component = pynutil.insert("day: \"") + cardinal + day_suffix + pynutil.insert("\"")

        graph_component = year_component | month_component

        graph_date = (
            year_component
            | month_component
            | (year_component + between_fields + month_component)
            | (month_component + between_fields + day_component)
            | (year_component + between_fields + month_component + between_fields + day_component)
        )

        final_graph = graph_component | graph_date

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
