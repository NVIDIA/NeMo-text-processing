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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_NARROW_NON_BREAK_SPACE,
    NEMO_NON_BREAKING_SPACE,
    GraphFst,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        2024/01/30 -> date { year: "二千二十四" month: "一" day: "三十" }
        2024/1/30 -> date { year: "二千二十四" month: "一" day: "三十" }
        2024-01-30 -> date { year: "二千二十四" month: "一" day: "三十" }
        2024.01.30 -> date { year: "二千二十四" month: "一" day: "三十" }
        H.6 -> date { era: "平成" yera: "六年" }
        R.1 -> date { era: "令和" "year: "元年" }
        S.5 -> date { era: "昭和" "year: "五年" }
        T.5 -> date { era: "大正" "year: "五年" }
        M.5 -> date { era: "明治" "year: "五年" }
        21日月曜日 -> tokens { date { day: "二十一日" weekday: "月曜日" } }
        70年代 -> date { year: "七十年代" }
        西暦794年 -> tokens { date { era: "西暦" year: "七百九十四年" } } 
        1月1日(月)〜3日(水) 
            -> tokens { date { month: "一月" day: "一日" weekday: "月曜日" } } tokens { name: "から" } tokens { date { day: "三日" weekday: "水曜日" } } 
        70〜80年代
            -> tokens { cardinal { integer: "七十" } } tokens { name: "から" } tokens { date { year: "八十年代" } }
        7月5〜9日(月〜金)
            -> tokens { date { month: "七月" } } tokens { cardinal { integer: "五" } } tokens { name: "から" } tokens { date { day: "九日" weekday: "月曜日" } } tokens { name: "から" } tokens { date { weekday: "金曜日" } } 
        7月初旬〜9月中旬
            -> tokens { date { month: "七月" } } tokens { name: "初" } tokens { name: "旬" } tokens { name: "から" } tokens { date { month: "九月" } } tokens { name: "中" } tokens { name: "旬" } 
        3〜4月
            -> tokens { cardinal { integer: "三" } } tokens { name: "から" } tokens { date { month: "四月" } }
        2023年3月1日(水)〜6月12日(火) 
            -> tokens { date { year: "二千二十三年" month: "三月" day: "一日" weekday: "水曜日" } } tokens { name: "から" } tokens { date { month: "六月" day: "十二日" weekday: "火曜日" } } 
        10月中旬〜11月上旬
            -> tokens { date { month: "十月" } } tokens { date { month: "中旬" } } tokens { name: "から" } tokens { date { month: "十一月" } } tokens { date { month: "上旬" } }
        1976年7月17日〜8月1日 
            -> tokens { date { year: "千九百七十六年" month: "七月" day: "十七日" } } tokens { name: "から" } tokens { date { month: "八月" day: "一日" } } 
    
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.just_cardinals

        month = pynutil.delete("0").ques + pynini.string_file(get_abs_path("data/date/month.tsv"))
        day = pynutil.delete("0").ques + pynini.string_file(get_abs_path("data/date/day.tsv"))
        week = pynini.string_file(get_abs_path("data/date/week.tsv"))
        era = pynini.string_file(get_abs_path("data/date/era.tsv"))
        era_abbrev = pynini.string_file(get_abs_path("data/date/era_abbrev.tsv"))

        signs = pynutil.delete("/") | pynutil.delete(".") | pynutil.delete("-")
        delete_spaces = pynini.closure(
            pynutil.delete(" ") | pynutil.delete(NEMO_NARROW_NON_BREAK_SPACE) | pynutil.delete(NEMO_NON_BREAKING_SPACE)
        )

        era_component = pynutil.insert("era: \"") + era + pynutil.insert("\"")
        era_abbrev_component = pynutil.insert("era: \"") + era_abbrev + pynutil.insert("\"")
        year_component = pynutil.insert("year: \"") + graph_cardinal + pynutil.insert("年") + pynutil.insert("\"")
        month_component = pynutil.insert("month: \"") + month + pynutil.insert("月") + pynutil.insert("\"")
        day_component = pynutil.insert("day: \"") + day + pynutil.insert("日") + pynutil.insert("\"")

        front_bracket = (
            (
                pynini.closure(pynutil.delete(delete_spaces))
                + pynutil.delete("(")
                + pynini.closure(pynutil.delete(delete_spaces))
            )
            | (
                pynini.closure(pynutil.delete(delete_spaces))
                + pynutil.delete("（")
                + pynini.closure(pynutil.delete(delete_spaces))
            )
            | (
                pynini.closure(pynutil.delete(delete_spaces))
                + pynutil.delete("（")
                + pynini.closure(pynutil.delete(delete_spaces))
            )
        )
        preceding_bracket = (
            (
                pynini.closure(pynutil.delete(delete_spaces))
                + pynutil.delete(")")
                + pynini.closure(pynutil.delete(delete_spaces))
            )
            | (
                pynini.closure(pynutil.delete(delete_spaces))
                + pynutil.delete("）")
                + pynini.closure(pynutil.delete(delete_spaces))
            )
            | (
                pynini.closure(pynutil.delete(delete_spaces))
                + pynutil.delete("）")
                + pynini.closure(pynutil.delete(delete_spaces))
            )
        )
        # this graph optionally accepts () around weekday to accomodate to inputs like (月〜金), thus being longer

        week_component = (
            (front_bracket + pynutil.insert("weekday: \"") + week + preceding_bracket + pynutil.insert("\""))
            | (
                front_bracket
                + pynutil.insert("weekday: \"")
                + week
                + pynini.cross("〜", "から")
                + week
                + preceding_bracket
                + pynutil.insert("\"")
            )
            | (
                front_bracket
                + pynutil.insert("weekday: \"")
                + week
                + pynutil.delete("・")
                + week
                + preceding_bracket
                + pynutil.insert("\"")
            )
        )

        # era, year, month, date
        graph_basic_date = (  # (R.|令和)2024/01/01, (R.|令和)2024/01/01/(水), (R.|令和)01/01, (R.|令和)01/01(水)
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynini.closure(year_component + signs + pynutil.insert(" "), 1)
            + month_component
            + signs
            + pynutil.insert(" ")
            + day_component
            + pynini.closure(pynutil.insert(" ") + week_component, 0, 1)
        )

        # 2024年, 9月, 28日
        individual_year_component = (
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynutil.insert("year: \"")
            + graph_cardinal
            + pynini.accep("年")
            + pynutil.insert("\"")
        )
        # this extra individual year component is to accomodate inputs R. 2024 with out "年"
        # the inputs may or maynot include "年", thus below:
        individual_year_component_2 = (
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynutil.insert("year: \"")
            + graph_cardinal
            + (pynini.accep("世紀") | pynini.accep(""))
            + pynutil.insert("\"")
        ) | (
            era_abbrev_component
            + pynutil.insert(" ")
            + pynutil.insert("year: \"")
            + graph_cardinal
            + pynutil.insert("年")
            + pynutil.insert("\"")
        )

        individual_month_component = (
            pynutil.insert("month: \"") + month + pynini.accep("月") + pynutil.insert("\"")
        ) | (
            pynutil.insert("month: \"")
            + (pynini.accep("中旬") | pynini.accep("下旬") | pynini.accep("上旬"))
            + pynutil.insert("\"")
        )
        individual_day_component = (
            pynutil.insert("day: \"") + graph_cardinal + pynini.accep("日") + pynutil.insert("\"")
        )

        graph_individual_component = (
            individual_year_component
            | individual_month_component
            | individual_day_component
            | week_component
            | individual_year_component_2
        ) + pynini.closure(pynutil.insert(" ") + week_component, 0, 1)

        # combined the above individual date components
        graph_individual_component_combined = (
            (individual_year_component + pynutil.insert(" ") + individual_month_component)
            | (individual_month_component + pynutil.insert(" ") + individual_day_component)
            | (
                individual_year_component
                + pynutil.insert(" ")
                + individual_month_component
                + pynutil.insert(" ")
                + individual_day_component
            )
        ) + pynini.closure(pynutil.insert(" ") + week_component, 0, 1)

        nendai = pynini.accep("年代")
        era_nendai = (
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynutil.insert("year: \"")
            + graph_cardinal
            + nendai
            + pynutil.insert("\"")
        )

        graph_all_date = (
            graph_basic_date | graph_individual_component | graph_individual_component_combined | era_nendai
        )

        final_graph = graph_all_date

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
