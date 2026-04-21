# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_ALL_NON_ZERO,
    NEMO_ALL_ZERO,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

days = pynini.string_file(get_abs_path("data/date/days.tsv"))
unambiguous_days = pynini.string_file(get_abs_path("data/date/unambiguous_days.tsv"))
months = pynini.string_file(get_abs_path("data/date/months.tsv"))
year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))
digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

digit_as_day = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))

with open(get_abs_path("data/date/suffixes.tsv"), "r", encoding="utf-8") as f:
    suffixes_list = [line.rstrip("\n") for line in f if line.strip()]
with open(get_abs_path("data/date/prefixes.tsv"), "r", encoding="utf-8") as f:
    prefixes_list = [line.rstrip("\n") for line in f if line.strip()]

suffix_union = pynini.union(*suffixes_list)
prefix_union = pynini.union(*prefixes_list)

verbalized_hundreds = teens_ties_hi.project("output")
verbalized_unit = pynini.union(
    teens_ties_hi.project("output"),
    digit.project("output")
)

verbalized_year_sou = (
    verbalized_hundreds
    + pynini.accep(" सौ")
    + pynini.closure(pynini.accep(" ") + verbalized_unit, 0, 1)
)


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "०१-०४-२०२४"         -> date { day: "एक" month: "अप्रैल" year: "दो हज़ार चौबीस" }
        "६ मार्च, २०१०"      -> date { day: "छह" month: "मार्च" year: "दो हज़ार दस" }
        "३१ मई, १९९० ई."    -> date { day: "इकतीस" month: "मई" year: "उन्नीस सौ नब्बे" era: "ईसवी" }
        "उन्नीस सौ बीस में"  -> date { era: "उन्नीस सौ बीस में" }
        "०३-२०१०"            -> date { month: "मार्च" year: "दो हज़ार दस" }
        "11-2024"             -> date { month: "नवंबर" year: "दो हज़ार चौबीस" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        # ── Year number graphs ────────────────────────────────────────────────
        graph_year_thousands = pynini.compose(
            (NEMO_ALL_DIGIT + NEMO_ALL_ZERO + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT),
            cardinal.graph_thousands
        )
        graph_year_hundreds_as_thousands = pynini.compose(
            (NEMO_ALL_DIGIT + NEMO_ALL_NON_ZERO + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT),
            cardinal.graph_hundreds_as_thousand
        )

        cardinal_graph = pynini.union(
            digit,
            teens_and_ties,
            cardinal.graph_hundreds,
            graph_year_thousands,
            graph_year_hundreds_as_thousands,
        )

        graph_year = pynini.union(graph_year_thousands, graph_year_hundreds_as_thousands)

        graph_year_era = pynini.union(
            graph_year_thousands,
            graph_year_hundreds_as_thousands,
            cardinal.graph_hundreds,
        )

        # ── Separators ───────────────────────────────────────────────────────
        delete_dash           = pynutil.delete("-")
        delete_slash          = pynutil.delete("/")
        delete_comma          = pynutil.delete(",")
        delete_space          = pynutil.delete(" ")
        delete_optional_space = pynini.closure(pynutil.delete(" "), 0, 1)
        delete_comma_sep      = delete_comma + delete_optional_space
        delete_numeric_sep    = pynini.union(delete_dash, delete_slash)

        # ── Day graphs ───────────────────────────────────────────────────────
        # Full day graph — all days 1-31 (used in DD-MM graphs)
        day_num = pynini.union(
            days,
            digit_as_day,
            teens_and_ties,
        )

        days_graph = (
            pynutil.insert("day: \"") + day_num + pynutil.insert("\"") + insert_space
        )

        # Unambiguous day graph — only days 13-31
        # Used in MM-DD graphs so they only fire when day cannot be a month number
        unambiguous_day_num = pynini.union(
            unambiguous_days,
        )

        unambiguous_days_graph = (
            pynutil.insert("day: \"") + unambiguous_day_num + pynutil.insert("\"") + insert_space
        )

        # ── Month graph ──────────────────────────────────────────────────────
        months_graph = (
            pynutil.insert("month: \"") + months + pynutil.insert("\"") + insert_space
        )

        # ── Year graph ───────────────────────────────────────────────────────
        years_graph = (
            pynutil.insert("year: \"") + graph_year + pynutil.insert("\"") + insert_space
        )

        # ── Era graph ────────────────────────────────────────────────────────
        era_graph = (
            pynutil.insert("era: \"") + year_suffix + pynutil.insert("\"") + insert_space
        )

        # ── Range graph (e.g. २९७-२७२ ई. पू.) ──────────────────────────────
        range_graph = pynini.cross("-", "से")

        # ── Century ordinal (e.g. २०वीं, १८वीं) ────────────────────────────
        century_number = (
            pynini.compose(pynini.closure(NEMO_ALL_DIGIT, 1), cardinal_graph)
            + pynini.accep("वीं")
        )
        century_text = (
            pynutil.insert("era: \"") + century_number + pynutil.insert("\"") + insert_space
        )

        # ── Year + suffix (e.g. २०२० में, १९९० का) ──────────────────────────
        year_number = graph_year + suffix_union
        year_text = (
            pynutil.insert("era: \"") + year_number + pynutil.insert("\"") + insert_space
        )

        # ── Year + prefix (e.g. सन् २०२४, साल २०२०) ────────────────────────
        year_prefix = (
            pynutil.insert("era: \"")
            + prefix_union
            + pynini.accep(" ")
            + graph_year
            + pynutil.insert("\"")
        )

        # ── Year + prefix + suffix (e.g. सन २००८ में) ───────────────────────
        year_prefix_suffix = (
            pynutil.insert("era: \"")
            + prefix_union
            + pynini.accep(" ")
            + graph_year
            + suffix_union
            + pynutil.insert("\"")
        )

        # ── Verbalized year passthrough graphs ───────────────────────────────
        graph_verbalized_year_suffix = (
            pynutil.insert("era: \"")
            + verbalized_year_sou
            + suffix_union
            + pynutil.insert("\"")
            + insert_space
        )

        graph_verbalized_year_bare = (
            pynutil.insert("era: \"")
            + verbalized_year_sou
            + pynutil.insert("\"")
            + insert_space
        )

        graph_verbalized_year_prefix = (
            pynutil.insert("era: \"")
            + prefix_union
            + pynini.accep(" ")
            + verbalized_year_sou
            + pynutil.insert("\"")
        )

        graph_verbalized_year_prefix_suffix = (
            pynutil.insert("era: \"")
            + prefix_union
            + pynini.accep(" ")
            + verbalized_year_sou
            + suffix_union
            + pynutil.insert("\"")
        )

        # ── Numeric separator date graphs ────────────────────────────────────
        # DD-MM: uses full day range (all 1-31)
        graph_dd_mm = days_graph + delete_numeric_sep + months_graph

        # MM-DD: only fires when day is unambiguously > 12
        # This prevents 01-10 being read as MM-DD (January 10)
        graph_mm_dd = months_graph + delete_numeric_sep + unambiguous_days_graph
        graph_mm_dd += pynutil.insert(" preserve_order: true ")

        # DD-MM-YYYY: uses full day range
        graph_dd_mm_yyyy = (
            days_graph
            + delete_numeric_sep
            + months_graph
            + delete_numeric_sep
            + years_graph
        )

        # MM-DD-YYYY: only fires when day is unambiguously > 12
        graph_mm_dd_yyyy = (
            months_graph
            + delete_numeric_sep
            + unambiguous_days_graph
            + delete_numeric_sep
            + years_graph
        )
        graph_mm_dd_yyyy += pynutil.insert(" preserve_order: true ")

        # ── Space-separated date graphs ──────────────────────────────────────
        graph_dd_month = (
            days_graph
            + delete_space
            + months_graph
        )

        graph_dd_month_comma_yyyy = (
            days_graph
            + delete_space
            + months_graph
            + delete_comma_sep
            + years_graph
        )

        graph_dd_month_comma_yyyy_era = (
            days_graph
            + delete_space
            + months_graph
            + delete_comma_sep
            + years_graph
            + era_graph
        )

        graph_month_comma_yyyy = (
            months_graph
            + delete_comma_sep
            + years_graph
        )

        graph_month_comma_yyyy_era = (
            months_graph
            + delete_comma_sep
            + years_graph
            + era_graph
        )

        # MM-YYYY: supports both space and dash separator
        # e.g. "मार्च २००३", "०३-२०१०", "11-2024"
        graph_mm_yyyy = (
            months_graph
            + pynini.union(delete_space, delete_dash)
            + years_graph
        )

        # ── Era-only graphs ──────────────────────────────────────────────────
        graph_year_era_only = (
            pynutil.insert("era: \"")
            + graph_year_era
            + insert_space
            + year_suffix
            + pynutil.insert("\"")
            + insert_space
        )

        graph_range = (
            pynutil.insert("era: \"")
            + cardinal_graph
            + insert_space
            + range_graph
            + insert_space
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" preserve_order: true ")
        )

        graph_year_suffix = era_graph

        # ── Final graph ───────────────────────────────────────────────────────
        final_graph = (
            # Full date with era — most specific first
            pynutil.add_weight(graph_dd_month_comma_yyyy_era, -0.003)
            | pynutil.add_weight(graph_month_comma_yyyy_era, -0.003)
            # Full numeric dates
            | pynutil.add_weight(graph_dd_mm_yyyy, -0.001)
            | graph_mm_dd_yyyy
            # Full space/comma dates
            | pynutil.add_weight(graph_dd_month_comma_yyyy, -0.001)
            # Day + month only
            | pynutil.add_weight(graph_dd_mm, -0.001)
            | pynutil.add_weight(graph_dd_month, -0.001)
            | graph_mm_dd
            # Month + year — space or dash
            | pynutil.add_weight(graph_mm_yyyy, -0.2)
            | pynutil.add_weight(graph_month_comma_yyyy, -0.2)
            # Era graphs
            | pynutil.add_weight(graph_year_era_only, -0.005)
            | pynutil.add_weight(graph_range, -0.005)
            | pynutil.add_weight(graph_year_suffix, -0.001)
            # Century ordinal
            | pynutil.add_weight(century_text, -0.001)
            # Verbalized year passthrough — more specific first
            | pynutil.add_weight(graph_verbalized_year_prefix_suffix, -0.012)
            | pynutil.add_weight(graph_verbalized_year_prefix, -0.011)
            | pynutil.add_weight(graph_verbalized_year_suffix, -0.010)
            | pynutil.add_weight(graph_verbalized_year_bare, -0.009)
            # Numeric year with suffix/prefix
            | pynutil.add_weight(year_prefix_suffix, -0.010)
            | pynutil.add_weight(year_prefix, -0.009)
            | pynutil.add_weight(year_text, -0.001)
        )

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph)