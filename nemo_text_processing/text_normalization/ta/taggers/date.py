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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.ta.graph_utils import (
    ASCII_TO_TA_DIGIT,
    NEMO_TA_DIGIT,
    NEMO_TA_ZERO,
    TO_TA_DIGITS,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying dates, e.g.
        15-06-2024 -> date { day: "பதினைந்து" month: "ஜூன்" year: "இரண்டாயிரத்து இருபத்துநான்கு" }
        2024-06-15 -> date { year: "இரண்டாயிரத்து இருபத்துநான்கு" month: "ஜூன்" day: "பதினைந்து" }
        15-06-2024ல் -> date { day: "பதினைந்து" month: "ஜூன்" year: "இரண்டாயிரத்து இருபத்துநான்கில்" }
        கி.பி. 2024 -> date { era: "கிறிஸ்து பிறகு" year: "இரண்டாயிரத்து இருபத்துநான்கு" }

    Reads ``data/date/days.tsv``, ``data/date/months.tsv`` and ``data/date/year_suffix.tsv``.
    A numeric date needs all three components with a 4-digit year and one separator
    throughout, so 15-06-24 and 10-20 are not dates.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        days = pynini.string_file(get_abs_path("data/date/days.tsv"))
        months = pynini.string_file(get_abs_path("data/date/months.tsv"))
        year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))

        # Two-digit day/month in either script; a single digit is zero-padded.
        pad_zero = pynutil.insert(NEMO_TA_ZERO)
        two_digit_input = pynini.union(
            NEMO_TA_DIGIT + NEMO_TA_DIGIT,
            pad_zero + NEMO_TA_DIGIT,
            pynini.compose(NEMO_DIGIT + NEMO_DIGIT, TO_TA_DIGITS),
            pad_zero + pynini.compose(NEMO_DIGIT, ASCII_TO_TA_DIGIT),
        ).optimize()
        days_graph = pynini.compose(two_digit_input, days).optimize()
        months_graph = pynini.compose(two_digit_input, months).optimize()

        # Four-digit years.
        year_graph = pynini.union(
            pynini.compose(NEMO_TA_DIGIT**4, cardinal.final_graph),
            pynini.compose(NEMO_DIGIT**4, cardinal.final_graph),
        ).optimize()

        delete_separator = pynutil.delete(pynini.union("-", "/", "."))

        # One date uses one separator throughout. That is enforced by filtering the input below
        # rather than by building each ordering once per separator, which would triple the
        # tagger; without it 15-06.2024 and 2024/06-15 also tag as dates.
        not_separator = pynini.difference(NEMO_CHAR, pynini.union("-", "/", "."))
        one_separator = pynini.union(
            *[
                pynini.closure(not_separator)
                + separator
                + pynini.closure(not_separator)
                + separator
                + pynini.closure(not_separator)
                for separator in ("-", "/", ".")
            ]
        ).optimize()

        day_component = pynutil.insert("day: \"") + days_graph + pynutil.insert("\"")
        month_component = pynutil.insert("month: \"") + months_graph + pynutil.insert("\"")
        # A case suffix or ordinal marker on the date lands on the year (2024ல், 2024க்கு, 2024ஆம்).
        year_component = (
            pynutil.insert("year: \"")
            + (year_graph | cardinal.attach_case_suffix(year_graph) | cardinal.ordinal_graph(year_graph))
            + pynutil.insert("\"")
        )

        graph_dd_mm_yyyy = (
            day_component
            + insert_space
            + delete_separator
            + month_component
            + insert_space
            + delete_separator
            + year_component
        )
        graph_mm_dd_yyyy = (
            month_component
            + insert_space
            + delete_separator
            + day_component
            + insert_space
            + delete_separator
            + year_component
            + pynutil.insert(" preserve_order: true")
        )
        graph_yyyy_mm_dd = (
            year_component
            + insert_space
            + delete_separator
            + month_component
            + insert_space
            + delete_separator
            + day_component
        )

        era_graph = pynutil.insert("era: \"") + year_suffix + pynutil.insert("\"")
        # The year after an era word is a date's year.
        era_graph |= era_graph + pynini.accep(" ") + pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")

        numeric_dates = pynini.compose(
            one_separator,
            pynutil.add_weight(graph_dd_mm_yyyy, -0.001)
            | pynutil.add_weight(graph_yyyy_mm_dd, -0.001)
            | graph_mm_dd_yyyy,
        )
        final_graph = numeric_dates | pynutil.add_weight(era_graph, -0.001)

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph)
