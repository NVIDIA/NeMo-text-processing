# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path, load_labels
from pynini.lib import pynutil

delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "01/11/2010" -> date { day: "الأول" month: "نوفمبر" year: "ألفين وعشرة" preserve_order: true }
        "1 نوفمبر" -> date { day: "الأول" month: "نوفمبر" }
        "نوفمبر 2010" -> date { month: "نوفمبر" year: "ألفين وعشرة" preserve_order: true }
        "2010" -> date { year: "ألفين وعشرة" }

    Supports the day-month-year, year-month-day, month-year, and bare-year orders,
    Gregorian and Hijri month names, and both numeric (with -, /, \\, . separators)
    and spelled-out month forms. The year is emitted in the idiomatic genitive form
    already produced by the cardinal grammar. Note that in the full pipeline a
    standalone number is classified as a cardinal, not a bare-year date.

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)
        months_name = pynini.string_file(get_abs_path("data/months/months_name.tsv")).optimize()

        month_abbr_graph = load_labels(get_abs_path("data/months/abbr_to_name.tsv"))
        number_to_month = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        number_to_month_hijri = pynini.string_file(get_abs_path("data/months/hijri_months.tsv")).optimize()
        self.month_hijri = number_to_month_hijri
        self.number_to_month = number_to_month
        month_graph = pynini.union(*[x[1] for x in month_abbr_graph]).optimize()
        month_abbr_graph = pynini.string_map(month_abbr_graph)
        month_abbr_graph = (
            pynutil.add_weight(month_abbr_graph, weight=0.0001) | (pynini.closure(NEMO_CHAR) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete(" ", weight=-0.0001), 0, 1)

        self.month_abbr = month_abbr_graph
        month_graph |= pynini.closure(NEMO_CHAR) @ month_graph
        # normalized name, or transliteration variant -> normalized name
        month_graph |= month_abbr_graph

        numbers = cardinal.graph
        ordinals = pynini.string_file(get_abs_path("data/ordinal/ordinals_date.tsv")).optimize()
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ ordinals
        day = (pynutil.insert("day: \"") + digit_day + pynutil.insert("\"")).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month
        number_to_month_hijri = digit_month @ number_to_month_hijri
        digit_month @= numbers

        month_name = (pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")).optimize()
        month_name_normalized = (pynutil.insert("month: \"") + months_name + pynutil.insert("\"")).optimize()
        month_number = (
            pynutil.insert("month: \"")
            + (pynutil.add_weight(digit_month, weight=0.0001) | number_to_month)
            + pynutil.insert("\"")
        ).optimize()
        month_number_hijri = (
            pynutil.insert("month: \"")
            + (pynutil.add_weight(digit_month, weight=0.0001) | number_to_month_hijri)
            + pynutil.insert("\"")
        ).optimize()

        # prefer cardinal over year
        year = pynutil.add_weight(numbers, weight=0.001)
        self.year = year

        year_only = pynutil.insert("year: \"") + year + pynutil.insert("\"")

        graph_dmy = (
            day
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + insert_space
            + month_name
            + pynini.closure(pynini.accep(" ") + year_only, 0, 1)
        )
        hijri_suffixes = pynini.string_file(get_abs_path("data/months/hijri_suffixes.tsv")).optimize()

        separators = ["-", "/", "\\"]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, " ") + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, " ") + month_number + year_optional
            self.year_hijri = year + pynini.accep(" ") + hijri_suffixes
            graph_year_hijri = pynutil.insert("year: \"") + self.year_hijri + pynutil.insert("\"")
            graph_dmy_hijri = day + pynini.cross(sep, " ") + month_number_hijri + pynini.cross(sep, " ") + graph_year_hijri
            graph_dmy |= new_graph
            graph_dmy |= graph_dmy_hijri

        # full day.month.year with a dot separator; the year is required so that
        # two-component decimals (e.g. "1.5") are not misread as a date
        year_only_4digit = pynutil.insert("year: \"") + ((NEMO_DIGIT ** 4) @ year) + pynutil.insert("\"")
        graph_dmy |= day + pynini.cross(".", " ") + month_number + pynini.cross(".", " ") + year_only

        # month + year, spelled ("نوفمبر 2010") and numeric ("12-2019");
        # "/" is intentionally excluded here because "11/2010" collides with the
        # fraction reading (11 على 2010), which wins the ambiguity
        graph_dmy |= month_name + pynini.accep(" ") + year_only
        graph_dmy |= month_number + pynini.cross("-", " ") + year_only_4digit

        dash = "-"
        day_optional = pynini.closure(pynini.cross(dash, " ") + day, 0, 1)
        graph_ymd = year_only + pynini.cross(dash, " ") + month_number + day_optional
        # 1 نوفمبر -> الأول من نوفمبر
        rule_1 = day + pynini.accep(" ") + month_name_normalized + pynini.closure(pynini.accep(" ") + year_only, 0, 1)

        # penalize the bare-year reading so a standalone number stays cardinal;
        # day/month/year dates below have no cardinal competitor and are unaffected
        final_graph = (
            (graph_dmy + pynutil.insert(" preserve_order: true"))
            | pynutil.add_weight(year_only, weight=1.0)
            | graph_ymd
            | rule_1
        )

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
