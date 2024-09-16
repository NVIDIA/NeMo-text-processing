# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path


class DateFst(GraphFst):
    def __init__(self):
        """
        Finite state transducer for classifying date
        1798年五月三十日 -> date { year: "1798" month: "5" day: "30" }
        五月三十日 -> date { month: "5" day: "30" }
        一六七二年 -> date { year: "1672" }
        """
        super().__init__(name="date", kind="classify")

        digits = pynini.string_file(get_abs_path("data/numbers/digit-nano.tsv"))  # imported for year-component
        months = pynini.string_file(get_abs_path("data/date/months.tsv"))  # imported for month-component
        days = pynini.string_file(get_abs_path("data/date/day.tsv"))  # imported for day-component

        # grammar for year
        graph_year = (
            pynini.closure(digits)
            + pynini.closure(pynini.cross("零", "0"))
            + pynini.closure(digits)
            + pynini.closure(pynini.cross("零", "0"))
            + pynutil.delete("年")
        )
        graph_year = pynutil.insert('year: "') + graph_year + pynutil.insert('"')

        # grammar for month
        graph_month = pynutil.insert('month: "') + months + pynutil.delete("月") + pynutil.insert('"')

        # grammar for day
        graph_day_suffix = pynini.accep("日") | pynini.accep("号") | pynini.accep("號")
        graph_delete_day_suffix = pynutil.delete(graph_day_suffix)
        graph_day = pynutil.insert('day: "') + days + graph_delete_day_suffix + pynutil.insert('"')

        # grammar for combinations of year+month, month+day, and year+month+day
        graph_ymd = graph_year + pynutil.insert(" ") + graph_month + pynutil.insert(" ") + graph_day
        graph_ym = graph_year + pynutil.insert(" ") + graph_month
        graph_md = graph_month + pynutil.insert(" ") + graph_day

        # final grammar for standard date
        graph_date = graph_ymd | graph_ym | graph_md | graph_year | graph_month | graph_day
        # graph_date = graph_year | graph_month | graph_day

        # grammar for optional prefix ad or bc
        graph_bc_prefix = pynini.closure("紀元前", 0, 1) | pynini.closure("公元前", 0, 1) | pynini.closure("纪元前", 0, 1)
        graph_bc = pynutil.delete(graph_bc_prefix)

        graph_ad_prefix = (
            pynini.closure("公元", 0, 1)
            | pynini.closure("公元后", 0, 1) + pynini.closure("紀元", 0, 1)
            | pynini.closure("纪元", 0, 1)
            | pynini.closure("西元", 0, 1)
        )
        graph_ad = pynutil.delete(graph_ad_prefix)

        graph_suffix_bc = (
            graph_bc + graph_date + pynutil.insert(' era: "') + pynutil.insert("B.C.") + pynutil.insert('"')
        )
        graph_suffix_ad = (
            graph_ad + graph_date + pynutil.insert(' era: "') + pynutil.insert("A.D.") + pynutil.insert('"')
        )

        graph_era = graph_suffix_bc | graph_suffix_ad

        # grammar for standard date and with era
        graph_date_final = graph_era | graph_date

        # graph_date_final = graph_date

        final_graph = self.add_tokens(graph_date_final)
        self.fst = final_graph.optimize()
