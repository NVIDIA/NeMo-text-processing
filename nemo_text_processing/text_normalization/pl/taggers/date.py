# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.pl.utils import get_abs_path


class DateFst(GraphFst):
    """Classifies Polish dates and exposes case-coordinated graphs."""

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        day_input = pynini.union(*(str(day) for day in range(1, 32)))
        numeric_day_input = day_input | pynini.union(*(f"{day:02d}" for day in range(1, 10)))

        month_number = pynini.string_file(get_abs_path("data/dates/months.tsv"))
        month_number = pynutil.delete("0") + month_number | month_number
        month_words = pynini.project(month_number, "output")
        month_abbr = pynini.string_file(get_abs_path("data/dates/month_abbr.tsv"))
        month_roman = pynini.string_file(get_abs_path("data/dates/months_roman.tsv"))

        year_prefix = ((NEMO_DIGIT - "0") + pynutil.insert("000")) @ cardinal.graphs["mi_sg_nom"]

        month_numeric_field = pynutil.insert(' month: "') + month_number + pynutil.insert('"')
        month_roman_field = pynutil.insert(' month: "') + month_roman + pynutil.insert('"')
        month_word_field = pynutil.insert(' month: "') + (month_words | month_abbr) + pynutil.insert('"')

        self.graphs = {}
        self.year_graphs = {}
        for slot, ordinal_graph in ordinal.graphs.items():
            if slot == "compound":
                continue
            day = day_input @ ordinal_graph
            numeric_day = numeric_day_input @ ordinal_graph
            year = year_prefix + insert_space + (NEMO_DIGIT**3 @ ordinal_graph)
            year_with_abbreviation = year + pynini.closure(
                pynini.closure(delete_space, 0, 1) + pynini.cross("r.", " roku"), 0, 1
            )
            day_field = pynutil.insert('day: "') + day + pynutil.insert('"')
            numeric_day_field = pynutil.insert('day: "') + numeric_day + pynutil.insert('"')
            year_field = pynutil.insert(' year: "') + year_with_abbreviation + pynutil.insert('"')

            numeric = pynini.union(
                *(
                    numeric_day_field
                    + pynutil.delete(separator)
                    + month_numeric_field
                    + pynutil.delete(separator)
                    + year_field
                    for separator in (".", "-", "/")
                )
            )
            numeric |= numeric_day_field + pynutil.delete(".") + month_roman_field + pynutil.delete(".") + year_field
            written = day_field + delete_space + month_word_field
            written += pynini.closure(delete_space + year_field, 0, 1)
            self.graphs[slot] = (numeric | written).optimize()
            self.year_graphs[slot] = (pynutil.insert('year: "') + year + pynutil.insert('"')).optimize()

        self.graph_dict = self.graphs
        if deterministic:
            self.final_graph = self.graphs["mi_sg_gen"]
        else:
            self.final_graph = pynini.union(*self.graphs.values(), *self.year_graphs.values()).optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
