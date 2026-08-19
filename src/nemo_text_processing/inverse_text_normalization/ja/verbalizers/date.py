# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.,
    date { day: "1" } -> 1日
    date { day: "5~9" } -> 5~9日
    date { month: "1" } -> 1月
    date { month: "3~4" } -> 3~4月
    date { month: "1" day: "1" } ->1月1日
    date { era: "70年代" } -> 70年代
    date { era: "70~80年代" } -> 70~80年代
    date { era: " 21世紀" } -> 21世紀
    date { year: "2009" } -> 2009年
    date { year: "23" month: "2" day: "25" weekday: "土" } -> 23年2月25日(土)
    date { month: "7" day: "5~9" weekday: "月~金" } -> 月5〜9日(月〜金)
    date { { year: "今年は令和6" } } -> 今年は令和6年
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        day_component = (
            pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("日") + pynutil.delete("\"")
        )
        month_component = (
            pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("月") + pynutil.delete("\"")
        )
        year_component = (
            pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("年") + pynutil.delete("\"")
        )
        era_component = pynutil.delete("era: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        weekday_component = pynutil.delete("weekday: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_regular = (
            pynini.closure(era_component)
            + pynini.closure(pynutil.delete(" ") + year_component, 0, 1)
            + pynini.closure(pynutil.delete(" ") + month_component, 0, 1)
            + pynini.closure(pynutil.delete(" ") + day_component, 0, 1)
            + pynini.closure(pynutil.delete(" ") + weekday_component, 0, 1)
        )
        graph = graph_regular | era_component

        final_graph = self.delete_tokens(graph)
        self.fst = final_graph.optimize()
