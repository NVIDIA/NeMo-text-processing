# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


def _get_month_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    return month_graph


def _get_year_graph(graph_two_digits, graph_thousands):
    """
    Transducer for year, e.g. twenty twenty -> 2020
    """
    graph_bc = pynini.string_file(get_abs_path("data/year_suffix.tsv")).invert()
    optional_graph_bc = pynini.closure(delete_space + insert_space + graph_bc, 0, 1)
    year_graph = pynini.union(
        (graph_two_digits + delete_space + graph_two_digits), #  # 20 19, 40 12, 20 20
        graph_thousands)  # 2012 - assuming no limit on the year
    year_graph += optional_graph_bc  # optional graph for BC
    year_graph.optimize()
    return year_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, 
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
        e.g. twenty twenty -> date { year: "2012" preserve_order: true }

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="date", kind="classify")

        ordinal_graph = ordinal.graph
        two_digits_graph = cardinal.graph_two_digit

        year_graph = _get_year_graph(two_digits_graph, cardinal.graph_thousands)
        year_graph = pynutil.add_weight(year_graph, 0.001)

        day_graph = pynutil.add_weight(two_digits_graph | ordinal_graph, -0.7)
        day_graph = pynutil.insert("day: \"") + day_graph + pynutil.insert("\"")

        month_graph = _get_month_graph()
        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")

        graph_year = (
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -0.001)
            + pynutil.insert("\"")
        )
        optional_graph_year = pynini.closure(graph_year, 0, 1,)

        optional_day_prefix = pynini.closure(pynutil.delete(pynini.union("ה", "ב")) + delete_space, 0, 1)
        month_prefix = pynutil.delete(pynini.union("ל", "ב"))
        graph_dmy = (
            optional_day_prefix
            + day_graph
            + delete_space
            + month_prefix
            + insert_space
            + month_graph
            + optional_graph_year
        )

        graph_my = (
            pynini.closure(month_prefix, 0, 1)
            + month_graph
            + graph_year
        )

        financial_period_graph = pynini.string_file(get_abs_path("data/date_period.tsv")).invert()
        period_fy = (
            pynutil.insert("text: \"")
            + financial_period_graph
            + (pynini.cross(" ", "") | pynini.cross(" ב", ""))
            + pynutil.insert("\"")
        )

        graph_year = (
            pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")
        )

        graph_fy = period_fy + pynutil.insert(" ") + graph_year

        final_graph = graph_dmy | graph_my | graph_year | graph_fy
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
    from nemo_text_processing.inverse_text_normalization.he.taggers.ordinal import OrdinalFst
    c = CardinalFst()
    g = DateFst(c, OrdinalFst(c)).fst
    apply_fst("הראשון ביוני אלפיים ושתיים עשרה", g)
    apply_fst("העשירי ביוני", g)
    apply_fst("מרץ אלף תשע מאות שמונים ותשע", g)
    apply_fst("שלושים למאי תשעים ותשע", g)
    apply_fst("שבעים לפני הספירה", g)
    apply_fst("עשרים עשרים ושתיים לפני הספירה", g)
    apply_fst("השלוש עשרה בינואר עשרים עשרים", g)
    apply_fst("בשני לפברואר עשרים עשרים ואחת", g)
    apply_fst("רבעון ראשון עשרים עשרים ושתיים", g)
    apply_fst("חציון שני בעשרים עשרים ושתיים", g)
