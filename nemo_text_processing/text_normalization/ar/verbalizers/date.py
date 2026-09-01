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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_preserve_order,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path, load_labels


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "الأول" month: "نوفمبر" year: "ألفين وعشرة" preserve_order: true }
            -> "الأول من شهر نوفمبر لعام ألفين وعشرة"
        date { month: "نوفمبر" year: "ألفين وعشرة" } -> "نوفمبر لعام ألفين وعشرة"
        date { year: "ألفين وعشرين" } -> "ألفين وعشرين"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day = pynutil.delete('day: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        months_names = pynini.union(*[x[1] for x in load_labels(get_abs_path("data/months/abbr_to_name.tsv"))])
        hijri_months_names = pynini.union(*[x[1] for x in load_labels(get_abs_path("data/months/hijri_months.tsv"))])
        month = pynutil.delete('month: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        final_month = month @ months_names
        final_month |= month @ hijri_months_names
        final_month |= month @ pynini.difference(NEMO_SIGMA, months_names | hijri_months_names)

        # The cardinal grammar already emits the idiomatic genitive year form
        # (e.g. 2020 -> "ألفين وعشرين"), so the year field passes through unchanged.
        year = pynutil.delete('year: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # day month year
        graph_dmy = (
            day
            + pynini.accep(" ")
            + pynutil.insert("من شهر ")
            + final_month
            + pynini.accep(" ")
            + pynutil.insert("لعام ")
            + year
        )
        graph_dmy |= final_month + pynini.accep(" ") + pynutil.insert("لعام ") + year

        # day month
        graph_dm = day + pynini.accep(" ") + pynutil.insert("من شهر ") + final_month

        self.graph = graph_dmy | pynutil.add_weight(year, weight=0.0001) | graph_dm
        final_graph = self.graph + delete_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
