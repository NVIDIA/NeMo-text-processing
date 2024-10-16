# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, in the form of (day) month (year) or year
        e.g. andra januari tjugohundraett -> tokens { name: "2001-01-02" }
        e.g. tjugotredje januari -> tokens { name: "23. jan." }
        e.g. tjugotjugo -> tokens { name: "2020" }

    Args:
        tn_date_tagger: TN date tagger
    """

    def __init__(
        self, tn_date_tagger: GraphFst,
    ):
        super().__init__(name="date", kind="classify")

        def force_double_digits(fst: GraphFst):
            double = (NEMO_DIGIT + NEMO_DIGIT) @ fst
            single = (pynutil.insert("0") + NEMO_DIGIT) @ (NEMO_DIGIT @ fst)
            return single | double

        year = tn_date_tagger.year.invert().optimize()
        decade = tn_date_tagger.decade.invert().optimize()
        era_words = tn_date_tagger.era_words.invert().optimize()
        day = tn_date_tagger.digit_day.invert().optimize()
        day_double = tn_date_tagger.digit_day_zero.invert().optimize()
        month_double = force_double_digits(tn_date_tagger.number_to_month).invert().optimize()
        month_abbr = tn_date_tagger.month_abbr.invert().optimize()
        self.month_to_number = tn_date_tagger.number_to_month.invert().optimize()

        graph_year = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        graph_month = pynutil.insert("month: \"") + month_double + pynutil.insert("\"")
        graph_month_abbr = pynutil.insert("month: \"") + month_abbr + pynutil.insert("\"")
        graph_day = pynutil.insert("day: \"") + day_double + pynutil.insert("\"")
        graph_day_ord = pynutil.insert("day: \"") + day + pynutil.insert("\"")
        graph_era = pynutil.insert("era: \"") + era_words + pynutil.insert("\"")
        optional_era = pynini.closure(NEMO_SPACE + graph_era, 0, 1)
        graph_decade = pynutil.insert("year: \"") + decade + pynutil.insert("\"")
        preserve = pynutil.insert(" preserve_order: true")
        optional_preserve = pynini.closure(preserve, 0, 1)

        year_era = graph_year + NEMO_SPACE + graph_era + preserve
        graph_dm = graph_day_ord + NEMO_SPACE + graph_month_abbr + preserve
        dmy = graph_day + NEMO_SPACE + graph_month + NEMO_SPACE + graph_year
        graph_dmy = dmy + optional_era
        ydm = graph_year + NEMO_SPACE + graph_month + NEMO_SPACE + graph_day
        graph_ydm = ydm + optional_era + preserve + optional_preserve
        final_graph = year_era | graph_dmy | graph_dm | graph_ydm | graph_decade

        graph = self.add_tokens(final_graph)
        self.fst = graph.optimize()
