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
    NEMO_SPACE,
    GraphFst,
)


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, in the form of (day) month (year) or year
        e.g. vierundzwanzigster Juli zwei tausend dreizehn -> date { day: "24." month: "Jul." year: "2013" }
        e.g. zwanzigzwanzig -> date { year: "2020" }
        e.g. vierzehnter Januar -> date { day: "14." month: "Jan." }
        e.g. zweiter dritter -> date { day: "2." month: "3." }
        e.g. Januar neunzehnachtzig -> date { month: "Jan." year: "1980" }
        e.g. circa zwei tausend vor Christus -> date { ca. year: "2000" era: "v. Ch." }
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="date", kind="classify")
        graph_ordinals = ordinal.graph_ordinals
        graph_first_dozen = ordinal.graph_ordinals_first_dozen
        graph_years = cardinal.graph_years
        graph_months_abbreviations = pynini.string_file(
            get_abs_path("/data/date/months.tsv")
        )

        # Graph for B.C and A.D
        vor = pynini.cross("vor", "v.")
        nach = pynini.cross("nach", "n.")
        christ = pynini.cross("christus", "chr.") | pynini.cross("Christus", "Chr.")
        BC_AD = (vor | nach) + pynini.accep(NEMO_SPACE) + christ

        # Accepts the formats:
        # day month (orthographic) year (e.g. fünfter Februar neunzehnhundertsechsundachtzig)
        # day month (numeric) year (e.g. fünfter zweiter neunzehnhundertschsundachtizg)
        # day month (e.g. fünfter Februar)
        # year (e.g. neunzehnhundertsechsundachtzig)
        # the above can be optionally followed by B.C or A.D
        day = graph_ordinals
        month = graph_months_abbreviations | graph_first_dozen
        year = graph_years

        graph_day_month = (
            pynutil.insert('day: "')
            + day
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert('month: "')
            + month
            + pynutil.insert('"')
        )
        graph_day_month_year = (
            pynutil.insert('day: "')
            + day
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert('month: "')
            + month
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + (pynutil.insert('year: "') + year + pynutil.insert('"'))
        )
        graph_year = pynutil.insert('year: "') + year + pynutil.insert('"')
        graph_date = graph_day_month | graph_day_month_year | graph_year

        # Adds support for the month year (e.g. Februar neunzehnhundertsechsundachtzig) format
        # N.B. the month needs to be "tied" to either the day or the year so as to prevent it from denormalizing when out of context
        graph_month_year = (
            pynutil.insert('month: "')
            + month
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert('year: "')
            + year
            + pynutil.insert('"')
        )

        graph_date |= graph_month_year

        # Adds BC or AD
        optional_era = (
            pynini.accep(NEMO_SPACE)
            + (pynutil.insert('era: "') + BC_AD + pynutil.insert('"'))
        ).ques

        graph_date += optional_era

        # Handles date-related terms (e.g. circa)
        circa = pynini.cross("circa", "ca.")
        suffix = pynini.accep("e").ques + pynini.union(*"nmrs").ques
        born = pynini.accep("geboren") + suffix.ques
        graph_born = pynini.cross(born, "geb.")
        date_terms = circa | graph_born

        # All years except for 1100-1999 when decontextualized will default to the incorrect ORDINAL formatting x.xxx
        # With the ORDINAL class weighted lower in the tokenizer, the system requires contextual cues to correctly transduce a four-digit sequence to a year
        year_cues = pynini.string_file(get_abs_path("/data/date/year_cues.tsv"))

        graph_date_terms = (
            pynutil.insert('text: "') + (date_terms | year_cues) + pynutil.insert('"')
        )
        graph_date = (graph_date_terms + pynini.accep(NEMO_SPACE)).ques + graph_date

        # Handles the decades e.g. die Sechziger (the 60s)
        # The graph utilizes Sparrowhawk's 'short_year' string field to handle the decades
        # The graph below removes "ein" as input for decades so as to prevent "einer" being tagged as a decade
        year_without_one = (
            pynini.difference(pynini.project(year, "input"), "ein")
        ) @ year
        decade_suffixes = pynini.accep("er") + pynini.accep("n").ques
        decades = year_without_one + decade_suffixes
        graph_decades = pynutil.insert('short_year: "') + decades + pynutil.insert('"')

        graph_date |= graph_decades

        # Generates the final graph
        graph = self.add_tokens(graph_date)
        self.fst = graph.optimize()
