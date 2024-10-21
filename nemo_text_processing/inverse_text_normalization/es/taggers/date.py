# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.es.graph_utils import int_to_roman
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. primero de enero -> date { day: "1" month: "enero" }
        e.g. uno de enero -> date { day: "1" month: "enero" }

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="date", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        graph_month = pynini.string_file(get_abs_path("data/dates/months.tsv"))
        graph_suffix = pynini.string_file(get_abs_path("data/dates/year_suffix.tsv")).invert()

        if input_case == INPUT_CASED:
            graph_month |= pynini.string_file(get_abs_path("data/dates/months_cased.tsv"))
            graph_suffix |= pynini.string_file(get_abs_path("data/dates/year_suffix_cased.tsv")).invert()

        graph_1_to_100 = pynini.union(
            graph_digit,
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + pynutil.delete(" y ") + graph_digit),
        )

        digits_1_to_31 = [str(digits) for digits in range(1, 32)]
        graph_1_to_31 = graph_1_to_100 @ pynini.union(*digits_1_to_31)
        # can use "primero" for 1st day of the month
        graph_1_to_31 = pynini.union(graph_1_to_31, pynini.cross("primero", "1"))

        day_graph = pynutil.insert("day: \"") + graph_1_to_31 + pynutil.insert("\"")

        month_graph = pynutil.insert("month: \"") + graph_month + pynutil.insert("\"")

        graph_dm = day_graph + delete_space + pynutil.delete("de") + delete_extra_space + month_graph

        # transform "siglo diez" -> "siglo x" and "año mil novecientos noventa y ocho" -> "año mcmxcviii"
        roman_numerals = int_to_roman(cardinal.graph)
        roman_centuries = pynini.union("siglo ", "año ") + roman_numerals
        roman_centuries_graph = pynutil.insert("year: \"") + roman_centuries + pynutil.insert("\"")

        # transform "doscientos antes de cristo" -> "200 a. c."
        year_with_suffix = cardinal.graph + pynini.accep(" ") + graph_suffix
        year_with_suffix_graph = pynutil.insert("year: \"") + year_with_suffix + pynutil.insert("\"")

        final_graph = graph_dm | roman_centuries_graph | year_with_suffix_graph
        final_graph += pynutil.insert(" preserve_order: true")

        if input_case == INPUT_CASED:
            final_graph |= capitalized_input_graph(final_graph)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
