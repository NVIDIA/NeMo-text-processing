# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.sv.graph_utils import SV_ALPHA
from nemo_text_processing.text_normalization.sv.utils import get_abs_path
from pynini.lib import pynutil

delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT
month_numbers = pynini.string_file(get_abs_path("data/dates/months.tsv"))
month_abbr = pynini.string_file(get_abs_path("data/dates/month_abbr.tsv"))
era_suffix = pynini.string_file(get_abs_path("data/dates/era_suffix.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "2:a januari, 2020" -> date { day: "andra" month: "januari" year: "tjugotjugotvå" }
        "2022.01.02" -> date { year: "tjugotjugotvå" month: "januari" day: "andra" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        number_to_month = month_numbers.optimize()
        self.month_abbr = month_abbr.optimize()
        month_graph = pynini.project(number_to_month, "output")

        numbers = cardinal.graph
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        optional_dot = pynini.closure(pynutil.delete("."), 0, 1)
        optional_comma = pynini.closure(pynutil.delete(","), 0, 1)

        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ ordinal.graph
        digit_words = pynini.project(digit_day, "output")
        day = (pynutil.insert("day: \"") + digit_day + optional_dot + pynutil.insert("\"")).optimize()
        day_sfx = (pynutil.insert("day: \"") + ordinal.suffixed_to_words + pynutil.insert("\"")).optimize()
        day_words = (pynutil.insert("day: \"") + digit_words + pynutil.insert("\"")).optimize()
        self.digit_day = digit_day

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month
        self.number_to_month = number_to_month

        month_name = (pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")).optimize()
        month_number = (pynutil.insert("month: \"") + number_to_month + pynutil.insert("\"")).optimize()
        month_abbreviation = (
            pynutil.insert("month: \"") + self.month_abbr + optional_dot + pynutil.insert("\"")
        ).optimize()

        # prefer cardinal over year
        year_first = ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0, 1)) @ numbers
        year_second = (
            pynini.union((NEMO_DIGIT - "0") + (NEMO_DIGIT - "0"), "0" + (NEMO_DIGIT - "0"), (NEMO_DIGIT - "0") + "0")
            @ numbers
        )
        year_second |= pynini.cross("00", "hundra")
        year_cardinal = ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 1, 3)) @ numbers
        year = pynini.union(year_first + year_second, year_first)  # 90, 990, 1990
        if not deterministic:
            year |= year_cardinal
        self.year = year

        year_second_decades = ((NEMO_DIGIT - "0") + "0") @ numbers
        year_second_decades |= pynini.cross("00", "hundra")
        decade_num = pynini.union(year_first + year_second_decades, year_second_decades)
        decade_word = pynini.union("tal", "talet", "tals")
        tals_word = "tals" + pynini.closure(SV_ALPHA, 1)
        tal_hyphen = pynutil.delete("-")
        if not deterministic:
            tal_hyphen |= pynini.cross("-", " ")
            decade_num |= ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 1, 2) + "0") @ numbers
        decade = (decade_num + tal_hyphen + (decade_word | tals_word)).optimize()
        # decade_only = pynutil.insert("decade: \"") + decade + pynutil.insert("\"")
        decade_only = pynutil.insert("year: \"") + decade + pynutil.insert("\"")

        year_only = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        era_only = pynutil.insert("era: \"") + era_suffix + pynutil.insert("\"")
        optional_era = pynini.closure(NEMO_SPACE + era_only, 0, 1)
        year_era = year_only + NEMO_SPACE + era_only + pynutil.insert(" preserve_order: true")
        year_opt_era = year_only + optional_era

        graph_dmy = (
            (day | day_sfx | day_words)
            + NEMO_SPACE
            + (month_name | month_abbreviation)
            + optional_comma
            + pynini.closure(NEMO_SPACE + year_opt_era, 0, 1)
        )

        day_optional = pynini.closure(pynini.cross("-", NEMO_SPACE) + day, 0, 1)
        graph_ymd = year_only + pynini.cross("-", NEMO_SPACE) + month_number + day_optional

        separators = [".", "-", "/"]
        for sep in separators:
            day_optional = pynini.closure(pynini.cross(sep, NEMO_SPACE) + day, 0, 1)
            year_optional = pynini.closure(pynini.cross(sep, NEMO_SPACE) + year_only + optional_era)
            new_graph = day + pynini.cross(sep, NEMO_SPACE) + month_number + year_optional
            graph_dmy |= new_graph
            graph_ymd |= year_only + pynini.cross(sep, NEMO_SPACE) + month_number + day_optional

        final_graph = graph_ymd | (graph_dmy + pynutil.insert(" preserve_order: true")) | year_era | decade_only

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
