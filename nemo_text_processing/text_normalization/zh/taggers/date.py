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

from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classfying dates, e.g.
        2002年       -> tokens { date { year: "二零零二" } }
        2002-01-28   -> tokens { date { year: "二零零二" month: "一" day: "二十八"} }
        2002/01/28   -> tokens { date { year: "二零零二" month: "一" day: "二十八"} }
        2002.01.28   -> tokens { date { year: "二零零二" month: "一" day: "二十八"} }
        2002年2月    -> tokens { date { year: "二零零二" month: "二" } }
        2月11日      -> tokens { date { month: "二" day: "十一" } }
        2002/02      -> is an error format according to the national standard
        02/11        -> is an error format according to the national standard
        According to national standard, only when the year, month, and day are all exist, it is allowed to use symbols to separate them

    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        month = pynini.string_file(get_abs_path("data/date/months.tsv"))
        day = pynini.string_file(get_abs_path("data/date/day.tsv"))
        suffix = pynini.string_file(get_abs_path("data/date/suffixes.tsv"))

        delete_sign = pynutil.delete('/') | pynutil.delete('-') | pynutil.delete('.') | pynutil.delete('·')
        delete_day = pynutil.delete('号') | pynutil.delete('號') | pynutil.delete('日')

        # grammar for only year, month, or day
        # atleast accep two digit to distinguish from year used for time
        # don't accept 日 to distinguish from day used fro time
        only_year = (
            pynutil.insert("year: \"")
            + pynini.closure(graph_digit | graph_zero, 2)
            + pynutil.delete('年')
            + pynutil.insert("\"")
        )
        only_month = pynutil.insert("month: \"") + month + pynutil.delete('月') + pynutil.insert("\"")
        only_day = pynutil.insert("day: \"") + day + delete_day + pynutil.insert("\"")
        # gh_1
        graph_only_date = only_year | only_month | only_day

        year_month = only_year + pynutil.insert(' ') + only_month
        month_day = only_month + pynutil.insert(' ') + only_day
        graph_ymd = only_year + pynutil.insert(' ') + only_month + pynutil.insert(' ') + only_day
        # gh_2
        graph_combination = year_month | month_day | graph_ymd

        year_component = (
            pynutil.insert("year: \"")
            + pynini.closure(graph_digit | graph_zero, 2, 4)
            + delete_sign
            + pynutil.insert("\"")
        )
        month_component = pynutil.insert("month: \"") + month + delete_sign + pynutil.insert("\"")
        day_component = pynutil.insert("day: \"") + day + pynutil.insert("\"")
        # gp_3
        graph_sign = year_component + pynutil.insert(' ') + month_component + pynutil.insert(' ') + day_component
        # gp_1+2+3
        graph_all = graph_only_date | graph_sign | graph_combination

        prefix = (
            pynini.accep('公元')
            | pynini.accep('西元')
            | pynini.accep('公元前')
            | pynini.accep('西元前')
            | pynini.accep('纪元')
            | pynini.accep('纪元前')
        )
        prefix_component = pynutil.insert("era: \"") + prefix + pynutil.insert("\"")
        # gp_prefix+(1,2,3)
        graph_prefix = prefix_component + pynutil.insert(' ') + (graph_ymd | year_month | only_year)

        suffix_component = pynutil.insert("era: \"") + suffix + pynutil.insert("\"")
        # gp_suffix +(1,2,3)
        graph_suffix = (graph_ymd | year_month | only_year) + pynutil.insert(' ') + suffix_component
        # gp_4
        graph_affix = graph_prefix | graph_suffix

        graph_suffix_year = (
            pynutil.insert("year: \"") + pynini.closure((graph_digit | graph_zero), 1) + pynutil.insert("\"")
        )
        graph_suffix_year = graph_suffix_year + pynutil.insert(' ') + suffix_component

        graph_with_era = graph_suffix_year | graph_affix

        graph = graph_only_date | graph_combination | graph_sign | graph_with_era

        # range
        symbol = pynini.accep("-") | pynini.accep("~") | pynini.accep("——") | pynini.accep("—")
        range_source = pynutil.insert("range: \"") + pynini.closure("从", 0, 1) + pynutil.insert("\"")
        range_goal = (
            pynutil.insert("range: \"")
            + (pynini.closure("到", 0, 1) | pynini.closure("至", 0, 1) | symbol)
            + pynutil.insert("\"")
        )
        graph_source = (
            range_source + pynutil.insert(' ') + graph + pynutil.insert(' ') + range_goal + pynutil.insert(' ') + graph
        )
        graph_goal = graph + pynutil.insert(' ') + range_goal + pynutil.insert(' ') + graph

        graph_range_final = graph_source | graph_goal

        final_graph = pynutil.add_weight(graph, -2.0) | graph_range_final

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
