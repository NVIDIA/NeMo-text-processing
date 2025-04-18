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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_ALPHA,
    NEMO_DIGIT,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)

graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()


def _get_month_graph(input_case: str = INPUT_LOWER_CASED):
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    if input_case == INPUT_CASED:
        month_graph |= pynini.string_file(get_abs_path("data/months_cased.tsv"))
    return month_graph


def _get_ties_graph(input_case: str):
    """
    Transducer for 20-99 e.g
    twenty three -> 23
    """
    graph = ties_graph + (delete_space + graph_digit | pynutil.insert("0"))
    if input_case == INPUT_CASED:
        graph = capitalized_input_graph(graph)
    return graph


def _get_range_graph(input_case: str):
    """
    Transducer for decades (1**0s, 2**0s), centuries (2*00s, 1*00s), millennia (2000s)
    """
    graph_ties = _get_ties_graph(input_case=input_case)
    graph = (graph_ties | graph_teen) + delete_space + pynini.cross("hundreds", "00s")
    graph |= pynini.cross("two", "2") + delete_space + pynini.cross("thousands", "000s")
    graph |= (
        (graph_ties | graph_teen)
        + delete_space
        + (pynini.closure(NEMO_ALPHA, 1) + (pynini.cross("ies", "y") | pynutil.delete("s")))
        @ (graph_ties | pynini.cross("ten", "10"))
        + pynutil.insert("s")
    )
    graph @= pynini.union("1", "2") + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT + "s"
    graph = capitalized_input_graph(graph)
    return graph


def _get_year_graph(input_case: str):
    """
    Transducer for year, e.g. twenty twenty -> 2020
    """

    def _get_digits_graph():
        zero = pynini.cross((pynini.accep("oh") | pynini.accep("o")), "0")
        graph = zero + delete_space + graph_digit
        graph.optimize()
        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph)
        return graph

    def _get_thousands_graph():
        graph_ties = _get_ties_graph(input_case)
        graph_hundred_component = (graph_digit + delete_space + pynutil.delete("hundred")) | pynutil.insert("0")
        optional_end = pynini.closure(pynutil.delete("and "), 0, 1)
        graph = (
            graph_digit
            + delete_space
            + pynutil.delete("thousand")
            + delete_space
            + graph_hundred_component
            + delete_space
            + (graph_teen | graph_ties | (optional_end + pynutil.insert("0") + graph_digit))
        )

        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph)
        return graph

    graph_ties = _get_ties_graph(input_case=input_case)
    graph_digits = _get_digits_graph()
    graph_thousands = _get_thousands_graph()
    graph_ad_bc = delete_space + pynini.string_file(get_abs_path("data/year_suffix.tsv")).invert()

    year_graph = (
        # 20 19, 40 12, 2012 - assuming no limit on the year
        (graph_teen + delete_space + (graph_ties | graph_digits | graph_teen))
        | (graph_ties + delete_space + (graph_ties | graph_digits | graph_teen))
        | graph_thousands
        | ((graph_digit + delete_space + (graph_ties | graph_digits | graph_teen)) + graph_ad_bc)
        | ((graph_digit | graph_teen | graph_digits | graph_ties | graph_thousands) + delete_space + graph_ad_bc)
        | ((graph_ties + delete_space + (graph_ties | graph_digits | graph_teen)) + delete_space + graph_ad_bc)
        | (
            (
                (graph_teen | graph_digit)
                + delete_space
                + pynutil.delete("hundred")
                + pynutil.insert("0")
                + (graph_digit | pynutil.insert("0"))
            )
            + delete_space
            + graph_ad_bc
        )
    )
    year_graph.optimize()
    if input_case == INPUT_CASED:
        year_graph = capitalized_input_graph(year_graph)
    return year_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
        e.g. twenty twenty -> date { year: "2012" preserve_order: true }

    Args:
        ordinal: OrdinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, ordinal: GraphFst, input_case: str):
        super().__init__(name="date", kind="classify")

        ordinal_graph = ordinal.graph
        year_graph = _get_year_graph(input_case=input_case)
        YEAR_WEIGHT = 0.001
        year_graph = pynutil.add_weight(year_graph, YEAR_WEIGHT)
        month_graph = _get_month_graph(input_case=input_case)

        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")

        day_graph = pynutil.insert("day: \"") + pynutil.add_weight(ordinal_graph, -0.7) + pynutil.insert("\"")
        graph_year = (
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert("\"")
        )
        optional_graph_year = pynini.closure(
            graph_year,
            0,
            1,
        )
        graph_mdy = month_graph + (
            (delete_extra_space + day_graph) | graph_year | (delete_extra_space + day_graph + graph_year)
        )
        the_graph = pynutil.delete("the")
        if input_case == INPUT_CASED:
            the_graph |= pynutil.delete("The").optimize()

        graph_dmy = (
            the_graph
            + delete_space
            + day_graph
            + delete_space
            + pynutil.delete("of")
            + delete_extra_space
            + month_graph
            + optional_graph_year
        )

        financial_period_graph = pynini.string_file(get_abs_path("data/date_period.tsv")).invert()
        period_fy = (
            pynutil.insert("text: \"")
            + financial_period_graph
            + (pynini.cross(" ", "") | pynini.cross(" of ", ""))
            + pynutil.insert("\"")
        )

        graph_year = (
            pynutil.insert("year: \"") + (year_graph | _get_range_graph(input_case=input_case)) + pynutil.insert("\"")
        )

        graph_fy = period_fy + pynutil.insert(" ") + graph_year

        final_graph = graph_mdy | graph_dmy | graph_year | graph_fy
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
