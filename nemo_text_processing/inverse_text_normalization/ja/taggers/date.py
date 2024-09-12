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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ja.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.,
    一日 -> 1日 date { day: "1" }
    五から九日 -> (5~9日) date { day: "5~9" }
    一月 -> 1月 date { month: "1" }
    三から四月 -> 3~4月 date { month: "3~4" }
    一月一日 -> 1月1日 date { month: "1" day: "1" }
    七十年代 -> 70年代 date {era: "70年代" }
    七十から八十年代 -> 70~80年代 date { era: "70~80年代" }
    二十一世紀 -> 21世紀 date { era: " 21世紀" }
    二千九年 -> 2009年 date { year: "2009" }
    二十三年二月二十五日土曜日~23年2月25日(土) -> { year: "23" month: "2" day: "25" weekday: "土" }
    七月五から九日月曜日から金曜日~7月5〜9日(月〜金) -> { month: "7" day: "5~9" weekday: "月~金" }
    今年はR六 -> { year: "今年は令和6" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        cardinal = cardinal.just_cardinals
        week = pynini.string_file(get_abs_path("data/date.tsv"))
        day = pynini.string_file(get_abs_path("data/day.tsv"))
        month = pynini.string_file(get_abs_path("data/months.tsv"))

        day_component = (
            pynutil.insert("day: \"")
            + cardinal
            + pynini.closure((pynini.cross("から", "〜") + day), 0, 1)
            + pynutil.delete("日")
            + pynutil.insert("\"")
        )
        month_component = (
            pynutil.insert("month: \"")
            + cardinal
            + pynini.closure((pynini.cross("から", "〜") + month), 0, 1)
            + pynutil.delete("月")
            + pynutil.insert("\"")
        )
        year_component = (
            pynutil.insert("year: \"")
            + cardinal
            + pynini.closure((pynini.cross("から", "〜") + cardinal), 0, 1)
            + pynutil.delete("年")
            + pynutil.insert("\"")
        )
        week_component = (pynutil.insert("weekday: \"(") + week + pynutil.insert(")\"")) | (
            pynutil.insert("weekday: \"(") + week + pynini.cross("から", "〜") + week + pynutil.insert(")\"")
        )
        graph_era = (
            pynutil.insert("era: \"")
            + cardinal
            + pynini.closure((pynini.cross("から", "〜") + cardinal), 0, 1)
            + (pynini.accep("年代") | pynini.accep("世紀"))
            + pynutil.insert("\"")
        )

        graph_component = day_component | month_component | year_component | graph_era | week_component
        graph_date = (
            pynini.closure(graph_era, 0, 1)
            + pynini.closure(year_component, 0, 1)
            + pynini.closure(pynutil.insert(" ") + month_component, 0, 1)
            + pynini.closure(pynutil.insert(" ") + day_component, 0, 1)
        )
        graph_date = graph_date | (graph_date + pynini.closure(pynutil.insert(" ") + week_component, 0, 1))

        # specific context for era year, e.g., L6 -> "令和6年"
        context = pynini.union(
            pynini.accep("今年は"),
            pynini.accep("来年は"),
            pynini.accep("再来年は"),
            pynini.accep("去年は"),
            pynini.accep("一昨年は"),
            pynini.accep("おととしは"),
        )
        era_year = pynini.union(
            pynini.cross("R", "令和"),
            pynini.cross("H", "平成"),
            pynini.cross("S", "昭和"),
            pynini.cross("T", "大正"),
            pynini.cross("M", "明治"),
        )
        graph_era_year = context + era_year + cardinal
        graph_content_specific = pynutil.insert("year: \"") + graph_era_year + pynutil.insert("\"")

        final_graph = graph_component | graph_date | graph_content_specific

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
