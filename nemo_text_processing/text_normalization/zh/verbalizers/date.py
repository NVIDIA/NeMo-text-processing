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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DateFst(GraphFst):
    """
    Finite state transducer classifiying dates, e.g.
    { date { year: "二零零二" } } -> 二零零二年
    { date { year: "二零零二" month: "一" day: "二十八"} } -> 二零零二年一月二十八日
    { date { year: "二零零二" month: "二" } } -> 二零零二年二月
    { date { month: "二" day: "十一" } } -> 二月十一日
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        year_component = (
            pynutil.delete("year: ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + pynutil.insert("年")
        )
        month_component = (
            pynutil.delete("month: ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + pynutil.insert("月")
        )
        day_component = (
            pynutil.delete("day: ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + pynutil.insert("日")
        )

        optional_era = (
            pynutil.delete("era: ") + pynutil.delete("\"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        )

        graph_date = (
            pynini.closure(year_component)
            + pynini.closure(delete_space)
            + pynini.closure(month_component)
            + pynini.closure(delete_space)
            + pynini.closure(day_component)
        )

        graph_date_era = pynini.union(
            (optional_era + delete_space + year_component),
            (optional_era + delete_space + year_component + delete_space + month_component),
            (
                optional_era
                + delete_space
                + year_component
                + delete_space
                + month_component
                + delete_space
                + day_component
            ),
        )

        graph_date_all = graph_date | graph_date_era

        # range
        symbol = pynini.accep("-") | pynini.accep("~") | pynini.accep("——") | pynini.accep("—")
        ranges = (
            pynutil.delete("range: \"")
            + delete_space
            + (pynini.closure(NEMO_NOT_QUOTE) | pynini.cross(symbol, "到"))
            + pynutil.delete("\"")
        )
        graph_range = (
            pynini.closure((ranges + delete_space), 0, 1)
            + graph_date
            + delete_space
            + ranges
            + delete_space
            + graph_date
        )

        final_graph = graph_date_all | graph_range
        # final_graph = optional_era + delete_space + year_component

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
