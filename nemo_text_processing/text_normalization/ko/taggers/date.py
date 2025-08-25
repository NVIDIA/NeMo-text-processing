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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst, insert_space, delete_space
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
        
        strip0 = pynini.closure(pynutil.delete("0"), 0, 1)
        graph_cardinal = cardinal.graph
        cardinal_lz = (strip0 + graph_cardinal).optimize()
        
        # Load base .tsv files
        week = pynini.string_file(get_abs_path("data/date/week.tsv"))
        month_exceptions = pynini.string_file(get_abs_path("data/date/exceptions.tsv"))
        month_exceptions_inputs = pynini.project(month_exceptions, "input").optimize()
        
        # Non-exception inputs go through the generic cardinal path
        graph_cardinal_non_exceptions = pynini.compose(
            pynini.difference(
                pynini.project(graph_cardinal, "input"), month_exceptions_inputs
            ).optimize(),
            graph_cardinal,
        ).optimize()
        
        # Month cardinal: prefer exceptions;
        month_cardinal = strip0 + (month_exceptions | graph_cardinal_non_exceptions).optimize()
                
        era = pynini.union("기원전", "기원후").optimize()
        signs = pynutil.delete("/") | pynutil.delete(".") | pynutil.delete("-")

        # Strict digit ranges for M/D/Y and Y/M/D
        _d = pynini.union(*[pynini.accep(str(i)) for i in range(10)])
        _1to9 = pynini.union(*[pynini.accep(str(i)) for i in range(1, 10)])

        # MM: 01-09 | 10-12
        MM = (pynini.accep("0") + _1to9) | (pynini.accep("1") + pynini.union("0", "1", "2"))
        
        # DD: 01-09 | 10-19 | 20-29 | 30-31
        DD = (
            (pynini.accep("0") + _1to9)
            | (pynini.accep("1") + _d)
            | (pynini.accep("2") + _d)
            | (pynini.accep("3") + pynini.union("0", "1"))
        )
        
        # YYYY: exactly 4 digits and two-digit year for M/D/YY and D/M/YY
        YYYY = _d + _d + _d + _d
        YY = _d + _d

        # Map digits -> cardinal words using existing graphs (strip leading zero via month_cardinal/cardinal_lz)
        mm_to_text = pynini.compose(MM, month_cardinal).optimize()
        dd_to_text = pynini.compose(DD, cardinal_lz).optimize()
        yyyy_to_text = pynini.compose(YYYY, graph_cardinal).optimize()
        yy_to_text = pynini.compose(YY, graph_cardinal).optimize()

        # Components with tags/suffixes (strict)
        month_component_md = (
            pynutil.insert("month: \"") + mm_to_text + pynutil.insert("월") + pynutil.insert("\"")
        ).optimize()
        day_component_md = (
            pynutil.insert("day: \"") + dd_to_text + pynutil.insert("일") + pynutil.insert("\"")
        ).optimize()
        year_component_y4 = (
            pynutil.insert("year: \"") + yyyy_to_text + pynutil.insert("년") + pynutil.insert("\"")
        ).optimize()
        year_component_y2 = (
            pynutil.insert("year: \"") + yy_to_text + pynutil.insert("년") + pynutil.insert("\"")
        ).optimize()

        # Prefer 4-digit year; still allow 2-digit with worse weight
        year_component_md = (year_component_y4 | pynutil.add_weight(year_component_y2, 1.0)).optimize()

        # Generic components
        era_component = pynutil.insert("era: \"") + era + pynutil.insert("\"")
        year_component = pynutil.insert("year: \"") + graph_cardinal + pynutil.insert("년") + pynutil.insert("\"")
        month_component = pynutil.insert("month: \"") + month_cardinal + pynutil.insert("월") + pynutil.insert("\"")
        day_component = pynutil.insert("day: \"") + cardinal_lz + pynutil.insert("일") + pynutil.insert("\"")

        # Brackets for weekday
        front_bracket = (
            pynini.closure(pynutil.delete(delete_space))
            + pynutil.delete("(")
            + pynini.closure(pynutil.delete(delete_space))
        ) | (
            pynini.closure(pynutil.delete(delete_space))
            + pynutil.delete("（")
            + pynini.closure(pynutil.delete(delete_space))
        )
        preceding_bracket = (
            pynini.closure(pynutil.delete(delete_space))
            + pynutil.delete(")")
            + pynini.closure(pynutil.delete(delete_space))
        ) | (
            pynini.closure(pynutil.delete(delete_space))
            + pynutil.delete("）")
            + pynini.closure(pynutil.delete(delete_space))
        )

        week_component_bracketed = (
            (front_bracket + pynutil.insert("weekday: \"") + week + preceding_bracket + pynutil.insert("\""))
            | (
                front_bracket
                + pynutil.insert("weekday: \"")
                + week
                + pynini.cross("〜", "부터")
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

        week_component_plain = pynutil.insert("weekday: \"") + week + pynutil.insert("\"")
        week_component = week_component_bracketed | week_component_plain

        # Format: YYYY/MM/DD(weekday)
        graph_basic_date = (
            pynini.closure(era_component + insert_space, 0, 1)
            + (pynutil.insert("year: \"") + graph_cardinal + pynutil.insert("년") + pynutil.insert("\""))
            + signs + insert_space
            + (pynutil.insert("month: \"") + month_cardinal + pynutil.insert("월") + pynutil.insert("\""))
            + signs + insert_space
            + (pynutil.insert("day: \"") + cardinal_lz + pynutil.insert("일") + pynutil.insert("\""))
            + pynini.closure(pynini.closure(insert_space, 0, 1) + week_component, 0, 1)
        )
        
        # American: MM/DD/YYYY
        graph_american_date = (
            month_component_md + signs + insert_space
            + day_component_md + signs + insert_space
            + year_component_md
            + pynini.closure(pynini.closure(insert_space, 0, 1) + week_component, 0, 1)
        ).optimize()

        # European: DD/MM/YYYY
        graph_european_date = (
            day_component_md + signs + insert_space
            + month_component_md + signs + insert_space
            + year_component_md
            + pynini.closure(pynini.closure(insert_space, 0, 1) + week_component, 0, 1)
        ).optimize()

        # Single elements (year/month/day)
        individual_year_component = (
            pynini.closure(era_component + insert_space, 0, 1)
            + pynutil.insert("year: \"")
            + graph_cardinal
            + pynutil.delete("년")
            + pynutil.insert("년")
            + pynutil.insert("\"")
        )

        individual_month_component = (
            pynutil.insert("month: \"")
            + month_cardinal
            + pynutil.delete("월")
            + pynutil.insert("월")
            + pynutil.insert("\"")
        )

        individual_day_component = (
            pynutil.insert("day: \"")
            + cardinal_lz
            + pynutil.delete("일")
            + pynutil.insert("일")
            + pynutil.insert("\"")
        )

        week_full_word_acceptor = pynini.project(week, "output")
        week_component_full_word = pynutil.insert("weekday: \"") + week_full_word_acceptor + pynutil.insert("\"")

        day_and_weekday_component = individual_day_component + pynini.closure(insert_space, 0, 1) + week_component_full_word

        month_and_weekday_component = individual_month_component + pynini.closure(insert_space, 0, 1) + week_component_full_word

        graph_individual_component = (
            day_and_weekday_component |   
            month_and_weekday_component |    
            individual_year_component | 
            individual_month_component | 
            individual_day_component | 
            week_component
        ) + pynini.closure(insert_space + week_component, 0, 1)

        graph_individual_component_combined = (
            (individual_year_component + insert_space + individual_month_component)
            | (individual_month_component + insert_space + individual_day_component)
            | (
                individual_year_component
                + insert_space
                + individual_month_component
                + insert_space
                + individual_day_component
            )
        ) + pynini.closure(insert_space + week_component, 0, 1)

        nendai = pynini.accep("년대")
        era_nendai = (
            pynini.closure(era_component + insert_space, 0, 1)
            + pynutil.insert("year: \"")
            + graph_cardinal
            + nendai
            + pynutil.insert("\"")
        ).optimize()

        graph_all_date = (
            graph_basic_date | graph_american_date | graph_european_date | graph_individual_component | graph_individual_component_combined | era_nendai
        ).optimize()

        final_graph = self.add_tokens(graph_all_date)
        self.fst = final_graph.optimize()
