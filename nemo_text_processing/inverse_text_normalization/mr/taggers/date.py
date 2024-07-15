# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.mr.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.inverse_text_normalization.mr.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite State Transducer for classifying dates
        e.g. दहा जानेवारी दोन हजार -> date { day: "१०" month: "जानेवारी" year: "२०००" preserve_order: true }
        e.g. इसवी सन दोन हजार बावीस -> date { text: "इ.स." year: "२०२२" preserve_order: true }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name='date', kind="classify")
        months = pynini.string_file(get_abs_path("data/date/months.tsv"))
        dates = pynini.string_file(get_abs_path("data/date/dates.tsv")).invert()
        prefixes = pynini.string_file(get_abs_path("data/date/prefixes.tsv"))

        YEAR_WEIGHT = 0.001
        month_graph = pynutil.insert("month: \"") + months + pynutil.insert("\" ")
        day_graph = pynutil.insert("day: \"") + dates + pynutil.insert("\" ")
        year_graph = cardinal.graph
        graph_year = (
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert("\"")
        )
        optional_graph_year = pynini.closure(graph_year, 0, 1,)
        graph_ad_bc = pynutil.insert("text: \"") + prefixes + delete_space + pynutil.insert("\"")

        graph_mdy = month_graph + (
            (delete_extra_space + day_graph) | graph_year | (delete_extra_space + day_graph + graph_year)
        )
        graph_dmy = day_graph + delete_space + month_graph + optional_graph_year
        graph_year_prefix = graph_ad_bc + graph_year

        final_graph = graph_mdy | graph_dmy | graph_year_prefix
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
