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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { year: "1798" month: "5" day: "30" } -> 1798年5月30日
        date { year: "1798" month: "5" } -> 1798年5月
        date { month: "5" day: "30" } -> 5月30日
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        # removing tokenization for year, month and day
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete('"')
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete('"')
        )
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete('"')
        )
        era = pynutil.delete("era:")
        bc = era + delete_space + pynutil.delete('"') + pynini.cross("A.D.", "公元") + pynutil.delete('"')
        ad = era + delete_space + pynutil.delete('"') + pynini.cross("B.C.", "公元前") + pynutil.delete('"')

        # combining above 3 for variations
        graph_ymd = (
            year
            + pynutil.insert("年")
            + delete_space
            + month
            + pynutil.insert("月")
            + delete_space
            + day
            + pynutil.insert("日")
        )
        graph_ym = year + pynutil.insert("年") + delete_space + month + pynutil.insert("月")
        graph_md = month + pynutil.insert("月") + delete_space + day + pynutil.insert("日")
        graph_year = year + pynutil.insert("年")
        graph_month = month + pynutil.insert("月")
        graph_day = day + pynutil.insert("日")
        graph_era = bc | ad

        optional_era = pynini.closure(graph_era)

        final_graph = (
            optional_era + delete_space + (graph_ymd | graph_ym | graph_md | graph_year | graph_month | graph_day)
        )

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
