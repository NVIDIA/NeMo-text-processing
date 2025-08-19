# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    NEMO_ALPHA,
    delete_space,
    GraphFst,
)


class DateFst(GraphFst):
    """
    WFST for verbalizing dates:

    e.g. date { day: "24." month: "Jul." year: "2013" } -> 24. Jul. 2013
    e.g. date { year: "2020" } -> 2020
    e.g. date { day: "14." month: "Jan." } -> 14. Jan.
    e.g. date { day: "2." month: "3." } -> 2. 3.
    e.g. date { month: "Jan." year: "1980" } -> Jan. 1980
    e.g. date { ca. year: "2000" era: "v. Ch." } -> ca. 2000 v. Ch.
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        period = pynini.accep(".")
        numeric_date_month = ((NEMO_DIGIT) | (NEMO_DIGIT**2)) + period
        numeric_day_month_year_leading_zero = (
            (pynutil.insert("0") + NEMO_DIGIT) | (NEMO_DIGIT**2)
        ) + period
        DE_char = pynini.union(*"äöüÄÖÜß").optimize()
        single_char = NEMO_ALPHA | DE_char
        month_abbreviation_acceptor = single_char**3 + period.ques
        year_acceptor = pynini.closure(NEMO_DIGIT, 1, 4)
        era_acceptor = (
            pynini.accep("v. chr.")
            | pynini.accep("v. Chr.")
            | pynini.accep("n. chr.")
            | pynini.accep("n. Chr.")
        )

        # Verbalizes the day segment
        # Two graphs are required due to formatting rules
        graph_day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + numeric_date_month
            + pynutil.delete('"')
        )

        graph_day_leading_zero = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + numeric_day_month_year_leading_zero
            + pynutil.delete('"')
        )

        # Verbalizes the month segment
        # Two graphs are required due to spacing and formatting rules

        graph_month_numeric_leading_zero = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + numeric_day_month_year_leading_zero
            + pynutil.delete('"')
        )

        graph_month_abbreviated = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + month_abbreviation_acceptor
            + pynutil.delete('"')
        )

        # Verbalizes the year segment
        graph_year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + year_acceptor
            + pynutil.delete('"')
        )

        # Verbalizes the era segment
        graph_era = (
            pynini.accep(NEMO_SPACE)
            + pynutil.delete("era:")
            + delete_space
            + pynutil.delete('"')
            + era_acceptor
            + pynutil.delete('"')
        )

        # Verbalizes date-related terms
        year_cues = pynini.string_file(get_abs_path("/data/date/year_cues.tsv"))
        circa_acceptor = pynini.accep("ca.")
        born_acceptor = pynini.accep("geb.")
        graph_date_terms = (
            pynutil.delete('text: "')
            + (circa_acceptor | born_acceptor | year_cues)
            + pynutil.delete('"')
            + pynini.accep(NEMO_SPACE)
        )

        graph_day_month_numeric = (
            graph_day_leading_zero
            + pynutil.delete(NEMO_SPACE)
            + graph_month_numeric_leading_zero
        )
        graph_day_month_abbreviated = (
            graph_day + pynini.accep(NEMO_SPACE) + graph_month_abbreviated
        )
        graph_day_month = graph_day_month_numeric | graph_day_month_abbreviated

        graph_month_year_numeric = (
            graph_month_numeric_leading_zero + pynutil.delete(NEMO_SPACE) + graph_year
        )
        graph_month_year_abbreviated = (
            graph_month_abbreviated + pynini.accep(NEMO_SPACE) + graph_year
        )
        graph_month_year = graph_month_year_numeric | graph_month_year_abbreviated

        graph_day_month_year_numeric = (
            graph_day_leading_zero
            + pynutil.delete(NEMO_SPACE).ques
            + graph_month_numeric_leading_zero
            + pynutil.delete(NEMO_SPACE).ques
            + graph_year
        )
        graph_day_month_year_abbreviated = (
            graph_day
            + pynini.accep(NEMO_SPACE)
            + graph_month_abbreviated
            + pynini.accep(NEMO_SPACE)
            + graph_year
        )
        graph_day_month_year = (
            graph_day_month_year_numeric | graph_day_month_year_abbreviated
        )

        graph_date = (
            graph_date_terms.ques
            + (graph_day_month | graph_month_year | graph_day_month_year | graph_year)
            + graph_era.ques
        ) | graph_date_terms

        # Handles the decades
        decade_suffixes = pynini.accep("er") + pynini.accep("n").ques
        graph_decades = (
            pynutil.delete('short_year: "')
            + year_acceptor
            + decade_suffixes
            + pynutil.delete('"')
        )

        graph_date |= graph_decades

        # Build and optimze the final graph
        delete_tokens = self.delete_tokens(graph_date)
        self.fst = delete_tokens.optimize()
