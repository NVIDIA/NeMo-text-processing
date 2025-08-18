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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ko.utils import get_abs_path

class DateFst(GraphFst):
    """
    Finite state transducer for classifying dates in Korean, e.g.
        2024/01/30 -> date { year: "이천이십사" month: "일월" day: "삼십" }
        2024/1/30 -> date { year: "이천이십사" month: "일월" day: "삼십" }
        2024-01-30 -> date { year: "이천이십사" month: "일월" day: "삼십" }
        2024.01.30 -> date { year: "이천이십사" month: "일월" day: "삼십" }

        기원전233년 -> date { era: "기원전" year: "이백삼십삼년" }
        기원후2024년 -> date { era: "기원후" year: "이천이십사년" }

        21일월요일 -> tokens { date { day: "이십일일" weekday: "월요일" } }
        1970년대 -> date { year: "천구백칠십년대" }

        1월1일(월)~3일(수)
            -> tokens { date { month: "일월" day: "일일" weekday: "월요일" } }
               tokens { name: "부터" }
               tokens { date { day: "삼일" weekday: "수요일" } }

        1970~1980년대
            -> tokens { cardinal { integer: "천구백칠십" } }
               tokens { name: "부터" }
               tokens { date { year: "천구백팔십년대" } }

        7월5~9일(월~금)
            -> tokens { date { month: "칠월" } }
               tokens { cardinal { integer: "오" } }
               tokens { name: "부터" }
               tokens { date { day: "구일" weekday: "월요일" } }
               tokens { name: "부터" }
               tokens { date { weekday: "금요일" } }

        2023년3월1일(수)~6월12일(화)
            -> tokens { date { year: "이천이십삼년" month: "삼월" day: "일일" weekday: "수요일" } }
               tokens { name: "부터" }
               tokens { date { month: "유월" day: "십이일" weekday: "화요일" } }
               
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.graph

        month = pynutil.delete("0").ques + pynini.string_file(get_abs_path("data/date/month_number.tsv"))
        day = pynutil.delete("0").ques + pynini.string_file(get_abs_path("data/date/day.tsv"))
        week = pynini.string_file(get_abs_path("data/date/week.tsv"))
        era = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))

        signs = pynutil.delete("/") | pynutil.delete(".") | pynutil.delete("-")
        delete_spaces = pynini.closure(pynutil.delete(" "))

        # This will match suffixes like "년"(year), "월"(month), "일"(day)
        era_component = pynutil.insert("era: \"") + era + pynutil.insert("\"")

        year_component = (
            pynutil.insert("year: \"") + graph_cardinal + pynutil.insert("년") + pynutil.insert("\"")
        )
        month_component = (
            pynutil.insert("month: \"") + month + pynutil.insert("월") + pynutil.insert("\"")
        )
        day_component = (
            pynutil.insert("day: \"") + day + pynutil.insert("일") + pynutil.insert("\"")
        )

        # Handle parentheses around weekdays
        front_bracket = (
            pynini.closure(pynutil.delete(delete_spaces)) + pynutil.delete("(") + pynini.closure(pynutil.delete(delete_spaces))
        ) | (
            pynini.closure(pynutil.delete(delete_spaces)) + pynutil.delete("（") + pynini.closure(pynutil.delete(delete_spaces))
        )
        preceding_bracket = (
            pynini.closure(pynutil.delete(delete_spaces)) + pynutil.delete(")") + pynini.closure(pynutil.delete(delete_spaces))
        ) | (
            pynini.closure(pynutil.delete(delete_spaces)) + pynutil.delete("）") + pynini.closure(pynutil.delete(delete_spaces))
        )

        week_component_bracketed = (
            front_bracket + pynutil.insert("weekday: \"") + week + preceding_bracket + pynutil.insert("\"")
        ) | (
            front_bracket + pynutil.insert("weekday: \"") + week + pynini.cross("〜", "부터") + week + preceding_bracket + pynutil.insert("\"")
        ) | (
            front_bracket + pynutil.insert("weekday: \"") + week + pynutil.delete("・") + week + preceding_bracket + pynutil.insert("\"")
        )
        
        week_component_plain = pynutil.insert("weekday: \"") + week + pynutil.insert("\"")
        week_component = week_component_bracketed | week_component_plain


        # Format: YYYY/MM/DD(weekday)
        graph_basic_date = (
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + (pynutil.insert("year: \"") + graph_cardinal + pynutil.insert("년") + pynutil.insert("\""))
            + signs + pynutil.insert(" ")
            + (pynutil.insert("month: \"") + month + pynutil.insert("월") + pynutil.insert("\""))
            + signs + pynutil.insert(" ")
            + (pynutil.insert("day: \"") + day + pynutil.insert("일") + pynutil.insert("\""))
            + pynini.closure(pynini.closure(pynutil.insert(" "), 0, 1) + week_component, 0, 1)
        )

        # Single elements (year/month/day)
        individual_year_component = (
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynutil.insert("year: \"") + graph_cardinal + pynutil.delete("년") + pynutil.insert("년") + pynutil.insert("\"")
        )
        individual_month_component = pynutil.insert("month: \"") + month + pynutil.delete("월") + pynutil.insert("월") + pynutil.insert("\"")
        individual_day_component   = pynutil.insert("day: \"")   + day   + pynutil.delete("일") + pynutil.insert("일") + pynutil.insert("\"")

        graph_individual_component = (
            individual_year_component | individual_month_component | individual_day_component | week_component
        ) + pynini.closure(pynutil.insert(" ") + week_component, 0, 1)

        graph_individual_component_combined = (
            (individual_year_component + pynutil.insert(" ") + individual_month_component)
            | (individual_month_component + pynutil.insert(" ") + individual_day_component)
            | (individual_year_component + pynutil.insert(" ") + individual_month_component + pynutil.insert(" ") + individual_day_component)
        ) + pynini.closure(pynutil.insert(" ") + week_component, 0, 1)
        
        nendai = pynini.accep("년대")
        era_nendai = (
            pynini.closure(era_component + pynutil.insert(" "), 0, 1)
            + pynutil.insert("year: \"")
            + graph_cardinal
            + nendai
            + pynutil.insert("\"")
        ).optimize()

        graph_all_date = (graph_basic_date | graph_individual_component | graph_individual_component_combined | era_nendai).optimize()

        final_graph = self.add_tokens(graph_all_date)
        self.fst = final_graph.optimize()