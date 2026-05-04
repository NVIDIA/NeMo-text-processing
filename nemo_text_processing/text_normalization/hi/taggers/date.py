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
months = pynini.string_file(get_abs_path("data/date/months.tsv"))
year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))
digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

with open(get_abs_path("data/date/suffixes.tsv"), "r", encoding="utf-8") as f:
    suffixes_list = [line.rstrip("\n") for line in f if line.strip()]
with open(get_abs_path("data/date/prefixes.tsv"), "r", encoding="utf-8") as f:
    prefixes_list = [line.rstrip("\n") for line in f if line.strip()]

suffix_union = pynini.union(*suffixes_list)
prefix_union = pynini.union(*prefixes_list)

verbalized_hundreds = teens_ties_hi.project("output")
verbalized_unit = pynini.union(verbalized_hundreds, digit.project("output"))

verbalized_year_sou = (
    verbalized_hundreds + pynini.accep(" सौ") + pynini.closure(pynini.accep(" ") + verbalized_unit, 0, 1)
)


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "०१-०४-२०२४" -> date { day: "एक" month: "अप्रैल" year: "दो हज़ार चौबीस" }
        "६ मार्च, २०१०" -> date { day: "छह" month: "मार्च" year: "दो हज़ार दस" }
        "३१ मई, १९९० ई." -> date { day: "इकतीस" month: "मई" year: "उन्नीस सौ नब्बे" era: "ईसवी" }
        "उन्नीस सौ बीस में" -> date { era: "उन्नीस सौ बीस में" }
        "०३-२०१०" -> date { month: "मार्च" year: "दो हज़ार दस" }
        "11-2024" -> date { month: "नवंबर" year: "दो हज़ार चौबीस" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        graph_year_thousands = pynini.compose(
            (NEMO_ALL_DIGIT + NEMO_ALL_ZERO + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT), cardinal.graph_thousands
        )
        graph_year_hundreds_as_thousands = pynini.compose(
            (NEMO_ALL_DIGIT + NEMO_ALL_NON_ZERO + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT), cardinal.graph_hundreds_as_thousand
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

        delete_dash = pynutil.delete("-")
        delete_slash = pynutil.delete("/")
        delete_comma = pynutil.delete(",")
        delete_space = pynutil.delete(" ")
        delete_optional_space = pynini.closure(pynutil.delete(" "), 0, 1)
        delete_comma_sep = delete_comma + delete_optional_space
        delete_numeric_sep = pynini.union(delete_dash, delete_slash)

        day_num = pynini.union(
            days,
            teens_and_ties,
        )

        days_graph = pynutil.insert("day: \"") + day_num + pynutil.insert("\"") + insert_space
        days_graph_with_year = days_graph

        unambiguous_ascii = pynini.union(*[str(i) for i in range(13, 32)])
        unambiguous_deva = pynini.union(
            "१३",
            "१४",
            "१५",
            "१६",
            "१७",
            "१८",
            "१९",
            "२०",
            "२१",
            "२२",
            "२३",
            "२४",
            "२५",
            "२६",
            "२७",
            "२८",
            "२९",
            "३०",
            "३१",
        )
        unambiguous_inputs = pynini.union(unambiguous_ascii, unambiguous_deva)
        unambiguous_day_num = pynini.compose(unambiguous_inputs, days)

        unambiguous_days_graph = pynutil.insert("day: \"") + unambiguous_day_num + pynutil.insert("\"") + insert_space

        month_name_acceptor = pynini.project(months, "output")

        months_numeric_fst = months

        months_graph_numeric = pynutil.insert("month: \"") + months_numeric_fst + pynutil.insert("\"") + insert_space

        months_fst = pynini.union(months_numeric_fst, month_name_acceptor)
        months_graph = pynutil.insert("month: \"") + months_fst + pynutil.insert("\"") + insert_space

        years_graph = pynutil.insert("year: \"") + graph_year + pynutil.insert("\"") + insert_space

        era_graph = pynutil.insert("era: \"") + year_suffix + pynutil.insert("\"") + insert_space

        range_graph = pynini.cross("-", "से")

        century_number = pynini.compose(pynini.closure(NEMO_ALL_DIGIT, 1), cardinal_graph) + pynini.accep("वीं")
        century_text = pynutil.insert("era: \"") + century_number + pynutil.insert("\"") + insert_space

        year_number = graph_year + suffix_union
        year_text = pynutil.insert("era: \"") + year_number + pynutil.insert("\"") + insert_space

        year_prefix = pynutil.insert("era: \"") + prefix_union + pynini.accep(" ") + graph_year + pynutil.insert("\"")

        year_prefix_suffix = (
            pynutil.insert("era: \"")
            + prefix_union
            + pynini.accep(" ")
            + graph_year
            + suffix_union
            + pynutil.insert("\"")
        )

        graph_verbalized_year_suffix = (
            pynutil.insert("era: \"") + verbalized_year_sou + suffix_union + pynutil.insert("\"") + insert_space
        )

        graph_verbalized_year_bare = (
            pynutil.insert("era: \"") + verbalized_year_sou + pynutil.insert("\"") + insert_space
        )

        graph_verbalized_year_prefix = (
            pynutil.insert("era: \"") + prefix_union + pynini.accep(" ") + verbalized_year_sou + pynutil.insert("\"")
        )

        graph_verbalized_year_prefix_suffix = (
            pynutil.insert("era: \"")
            + prefix_union
            + pynini.accep(" ")
            + verbalized_year_sou
            + suffix_union
            + pynutil.insert("\"")
        )

        graph_dd_mm = days_graph + delete_numeric_sep + months_graph

        graph_mm_dd = months_graph + delete_numeric_sep + unambiguous_days_graph
        graph_mm_dd += pynutil.insert(" preserve_order: true ")

        graph_dd_mm_yyyy = days_graph_with_year + delete_numeric_sep + months_graph + delete_numeric_sep + years_graph

        graph_mm_dd_yyyy = (
            months_graph + delete_numeric_sep + unambiguous_days_graph + delete_numeric_sep + years_graph
        )
        graph_mm_dd_yyyy += pynutil.insert(" preserve_order: true ")

        graph_dd_month = days_graph + delete_space + months_graph_numeric

        graph_dd_month_comma_yyyy = days_graph + delete_space + months_graph + delete_comma_sep + years_graph

        graph_dd_month_comma_yyyy_era = (
            days_graph + delete_space + months_graph + delete_comma_sep + years_graph + era_graph
        )

        graph_month_comma_yyyy = months_graph + delete_comma_sep + years_graph

        graph_month_comma_yyyy_era = months_graph + delete_comma_sep + years_graph + era_graph

        graph_mm_yyyy = months_graph + pynini.union(delete_space, delete_dash) + years_graph

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

        final_graph = (
            pynutil.add_weight(graph_dd_month_comma_yyyy_era, -0.003)
            | pynutil.add_weight(graph_month_comma_yyyy_era, -0.003)
            | pynutil.add_weight(graph_dd_mm_yyyy, -0.001)
            | graph_mm_dd_yyyy
            | pynutil.add_weight(graph_dd_month_comma_yyyy, -0.001)
            | pynutil.add_weight(graph_dd_mm, -0.001)
            | pynutil.add_weight(graph_dd_month, -0.001)
            | graph_mm_dd
            | pynutil.add_weight(graph_mm_yyyy, -0.2)
            | pynutil.add_weight(graph_month_comma_yyyy, -0.2)
            | pynutil.add_weight(graph_year_era_only, -0.005)
            | pynutil.add_weight(graph_range, -0.005)
            | pynutil.add_weight(graph_year_suffix, -0.001)
            | pynutil.add_weight(century_text, -0.001)
            | pynutil.add_weight(graph_verbalized_year_prefix_suffix, -0.012)
            | pynutil.add_weight(graph_verbalized_year_prefix, -0.011)
            | pynutil.add_weight(graph_verbalized_year_suffix, -0.010)
            | pynutil.add_weight(graph_verbalized_year_bare, -0.009)
            | pynutil.add_weight(year_prefix_suffix, -0.010)
            | pynutil.add_weight(year_prefix, -0.009)
            | pynutil.add_weight(year_text, -0.001)
        )

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph)
