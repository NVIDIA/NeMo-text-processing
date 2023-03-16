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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from pynini.lib import pynutil


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

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        optional_delete_space = pynini.closure(NEMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        def force_double_digits(fst: GraphFst):
            double = (NEMO_DIGIT + NEMO_DIGIT) @ fst
            single = (pynutil.insert("0") + NEMO_DIGIT) @ (NEMO_DIGIT @ fst)
            return (single | double)

        year = tn_date_tagger.year.invert().optimize()
        decade = tn_date_tagger.decade.invert().optimize()
        era_suffix = tn_date_tagger.era_suffix.invert().optimize()
        day = tn_date_tagger.digit_day.invert().optimize()
        day_double = force_double_digits(tn_date_tagger.digit_day).invert().optimize()
        month_double = force_double_digits(tn_date_tagger.number_to_month).invert().optimize()
        month_abbr = tn_date_tagger.month_abbr.invert().optimize()
        month = tn_date_tagger.number_to_month.invert().optimize()

        graph_year = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        graph_month = pynutil.insert("month: \"") + month + pynutil.insert("\"")
        graph_month_abbr = pynutil.insert("month: \"") + month_abbr + pynutil.insert("\"")
        graph_day = pynutil.insert("day: \"") + day_double + pynutil.insert("\"")
        graph_era = pynutil.insert("era: \"") + era_suffix + pynutil.insert("\"")
        graph_decade = pynutil.insert("year: \"") + decade + pynutil.insert("\"")

        graph = pynutil.insert("name: \"") + convert_space(final_graph) + pynutil.insert("\"")
        self.fst = graph.optimize()
