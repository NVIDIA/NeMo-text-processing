# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
)
from nemo_text_processing.text_normalization.sv.utils import get_abs_path

era_words = pynini.string_file(get_abs_path("data/dates/era_words.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "trettioförsta" month: "mars" year: "tjugotjugotvå" } -> "trettioförsta mars tjugotjugotvå"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        year = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        era = pynutil.delete("era: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        era_opt = pynini.closure(NEMO_SPACE + era, 0, 1)
        year_era_opt = year + era_opt

        # day month year
        graph_year_era = year + NEMO_SPACE + era + delete_preserve_order
        graph_year_era |= year + NEMO_SPACE + era
        graph_dmy = pynini.union(
            day + NEMO_SPACE + month + pynini.closure(NEMO_SPACE + year_era_opt, 0, 1) + delete_preserve_order
        )
        graph_was_ymd = pynini.union(month + NEMO_SPACE + year, day + NEMO_SPACE + month + NEMO_SPACE + year)

        self.graph = graph_dmy | graph_year_era | graph_was_ymd | year
        final_graph = self.graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
