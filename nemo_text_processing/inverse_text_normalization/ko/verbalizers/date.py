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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst, NEMO_SPACE


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. 이천십이년 일월 오일 -> date { year: "2012" month: "1" day: "5"  }
        e.g. 오월 -> date { month: "5" }
        e.g. 칠일 -> date { day: "7" }
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        year_component = (
            pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("년") + pynutil.delete("\"")
        )
        month_component = (
            pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("월") + pynutil.delete("\"")
        )
        day_component = (
            pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("일") + pynutil.delete("\"")
        )
        
        graph = (
            pynini.closure(pynutil.delete(NEMO_SPACE) + year_component, 0, 1)
            + pynini.closure(pynutil.delete(NEMO_SPACE) + month_component, 0, 1)
            + pynini.closure(pynutil.delete(NEMO_SPACE) + day_component, 0, 1)
        )

        final_graph = self.delete_tokens(graph)
        self.fst = final_graph.optimize()
