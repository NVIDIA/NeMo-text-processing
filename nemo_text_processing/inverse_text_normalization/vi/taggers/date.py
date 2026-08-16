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

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date,
        e.g. mười lăm tháng một năm hai nghìn mười hai -> date { day: "15" month: "1" year: "2012" preserve_order: true }
        e.g. ngày ba mốt tháng mười hai năm một chín chín chín -> date { day: "31" month: "12" year: "2012" preserve_order: true }
        e.g. năm hai không hai mốt -> date { year: "2021" preserve_order: true }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
        ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()

        # Special digit mappings for Vietnamese
        graph_one = pynini.cross("mốt", "1")
        graph_four = pynini.cross("tư", "4")
        graph_five = pynini.cross("lăm", "5")
        graph_ten = pynini.cross("mươi", "")
        optional_ten = pynini.closure(delete_space + graph_ten, 0, 1)

        # Ties graph for 20-99 (e.g., "hai ba" -> "23")
        graph_ties = pynini.union(
            ties_graph + optional_ten + delete_space + pynini.union(graph_digit, graph_one, graph_four, graph_five),
            ties_graph + delete_space + graph_ten + pynutil.insert("0", weight=0.01),
        )

        # Zero prefix patterns (e.g., "linh năm" -> "05")
        zero = pynini.cross((pynini.union("linh", "lẻ")), "0")
        graph_digits = pynini.union(
            zero + delete_space + pynini.union(graph_digit, graph_four),
            graph_zero + delete_space + graph_digit,
        ).optimize()

        # Year components
        # Hundreds pattern (e.g., "hai trăm mười hai" -> "212")
        year_hundreds = (
            graph_digit
            + delete_space
            + pynutil.delete("trăm")
            + delete_space
            + pynini.union(graph_teen, graph_ties, graph_digits)
        )

        # Thousands pattern with optional hundreds (e.g., "hai nghìn không ba" -> "2003")
        year_hundred_component = pynini.union(
            pynini.union(graph_digit, graph_zero) + delete_space + pynutil.delete("trăm"),
            pynutil.insert("0", weight=0.01),
        )
        year_thousands = (
            graph_digit
            + delete_space
            + pynutil.delete(pynini.union("nghìn", "ngàn"))
            + delete_space
            + year_hundred_component
            + delete_space
            + pynini.union(graph_teen, graph_ties, graph_digits)
        )

        # Standard XYYZ pattern (e.g., "hai không một chín" -> "2019")
        year_standard = (
            graph_digit
            + delete_space
            + pynini.union(graph_digit, graph_zero)
            + delete_space
            + pynini.union(graph_teen, graph_ties, graph_digits)
        )

        # XYZ pattern with implied 0 (e.g., "hai không hai mốt" -> "2021")
        year_implied_zero = (
            graph_digit
            + pynutil.insert("0", weight=0.01)
            + delete_space
            + pynini.union(graph_ties, graph_digits, graph_teen)
        )

        # Digit-by-digit pattern (e.g., "hai không một chín" -> "2019")
        year_digit_by_digit = (
            pynini.union(graph_digit, graph_zero)
            + delete_space
            + pynini.union(graph_digit, graph_zero)
            + delete_space
            + pynini.union(graph_digit, graph_zero)
            + delete_space
            + pynini.union(graph_digit, graph_zero)
        )

        year_graph = pynini.union(
            year_standard,
            year_thousands,
            year_hundreds,
            year_implied_zero,
            year_digit_by_digit,
        ).optimize()

        # Month graph with special handling for "năm" (means "5" in months but "year" in other contexts)
        month_graph = (
            pynutil.insert('month: "')
            + pynini.string_file(get_abs_path("data/months.tsv")).optimize()
            + pynutil.insert('"')
        )
        month_exception = pynini.project(pynini.cross("năm", "5"), "input")
        month_graph_exception = (pynini.project(month_graph, "input") - month_exception.arcsort()) @ month_graph

        day_graph = pynutil.insert('day: "') + cardinal_graph + pynutil.insert('"')
        graph_month = pynutil.delete("tháng") + delete_space + month_graph_exception

        graph_year = pynutil.add_weight(
            delete_extra_space
            + pynutil.delete("năm")
            + delete_extra_space
            + pynutil.insert('year: "')
            + year_graph
            + pynutil.insert('"'),
            -0.1,
        )

        # Date pattern combinations
        # Pattern 1: Day-Month-Year (e.g., "ngày 15 tháng 1 năm 2024")
        graph_dmy = (
            day_graph
            + delete_space
            + pynutil.delete("tháng")
            + delete_extra_space
            + month_graph
            + pynini.closure(graph_year, 0, 1)
        )

        # Pattern 2: Month-Year (e.g., "tháng 1 năm 2024")
        graph_my = pynutil.delete("tháng") + delete_space + month_graph + graph_year

        # Pattern 3: Standalone year (e.g., "năm 2024")
        graph_year_standalone = pynutil.add_weight(
            pynutil.delete("năm") + delete_extra_space + pynutil.insert('year: "') + year_graph + pynutil.insert('"'),
            -0.1,
        )

        final_graph = pynini.union(
            graph_dmy,  # Day-Month-Year
            graph_my,  # Month-Year
            graph_month,  # Month only
            graph_year_standalone,  # Year only
        ) + pynutil.insert(" preserve_order: true")

        self.fst = self.add_tokens(final_graph).optimize()
