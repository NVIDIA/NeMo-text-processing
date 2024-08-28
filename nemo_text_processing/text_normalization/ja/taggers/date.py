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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst
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
        1月1日(月)〜3日(水) -> 一月一日月曜日から三日水曜日 # combiend with punct grammar
        70〜80年代 -> 七十から八十年代
        70年代 -> 七十年代
        7月5〜9日(月〜金) -> 七月五から九日月曜日から金曜日
        7月初旬〜9月中旬 -> 七月初旬から九月中旬
        3〜4月 -> 三から四月
        21日月曜日 -> 二十一日月曜日
        2023年3月1日(水)〜6月12日(火) -> 二千二十三年三月一日水曜日から六月十二日火曜日
        10月中旬〜11月上旬 -> 十月中旬から十一月上旬
        1976年7月17日〜8月1日 -> 千九百七十六年七月十七日から八月一日
        西暦794年 -> 西暦七百九十四年

    
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.just_cardinals

        month = pynini.string_file(get_abs_path("data/date/month.tsv"))
        day = pynini.string_file(get_abs_path("data/date/day.tsv"))
        week = pynini.string_file(get_abs_path("data/date/week.tsv"))
        era = pynini.string_file(get_abs_path("data/date/era.tsv"))

        signs = pynutil.delete("/") | pynutil.delete(".") | pynutil.delete("-")
        words = pynini.accep("年") | pynini.accep("月") | pynini.accep("日")

        era_component = pynutil.insert("era: \"") + era + pynutil.insert("\"")
        year_component = pynutil.insert("year: \"") + graph_cardinal + pynutil.insert("年") + pynutil.insert("\"")
        month_component = pynutil.insert("month: \"") + month + pynutil.insert("月") + pynutil.insert("\"")
        day_component = pynutil.insert("day: \"") + day + pynutil.insert("日") + pynutil.insert("\"")

        front_bracket = (
            pynutil.delete("(")
            | pynutil.delete("（")
            | pynutil.delete("（") 
        )
        preceding_bracket = (
            pynutil.delete(")")
            | pynutil.delete("）")
            | pynutil.delete("）")
        )
        week_component = (
            front_bracket
            + pynutil.insert("weekday: \"")
            + week
            + preceding_bracket
            + pynutil.insert("\"")
        )

        # era, year, month, date
        graph_basic_date = (  # (R.|令和)2024/01/01, (R.|令和)2024/01/01/(水), (R.|令和)01/01, (R.|令和)01/01(水)
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynini.closure(year_component + signs + pynutil.insert(" "), 0, 1)
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
        individual_month_component = pynutil.insert("month: \"") + month + pynini.accep("月") + pynutil.insert("\"")
        individual_day_component = (
            pynutil.insert("day: \"") + graph_cardinal + pynini.accep("日") + pynutil.insert("\"")
        )

        graph_individual_component = (individual_year_component | individual_month_component | individual_day_component | week_component) + pynini.closure(pynutil.insert(" ") + week_component, 0, 1)
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

        graph_regular_date = graph_basic_date | graph_individual_component | graph_individual_component_combined


        final_graph = graph_regular_date 

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
