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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese (Brazilian) time, e.g.
        14:30 -> time { hours: "catorze" minutes: "trinta" preserve_order: true }
        14:30:05 -> time { hours: "catorze" minutes: "trinta" seconds: "cinco" preserve_order: true }
        12:00 -> time { hours: "doze" preserve_order: true }
        11:00 da manhã -> time { hours: "onze" suffix: "da manhã" preserve_order: true }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        )

        graph_hour = (
            delete_leading_zero_to_double_digit
            @ pynini.union(*labels_hour)
            @ cardinal_graph
        )

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal_graph
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal_graph
        final_graph_minute = (
            pynutil.insert('minutes: "')
            + (
                pynutil.delete("0") + graph_minute_single
                | graph_minute_double
            )
            + pynutil.insert('"')
        )

        final_graph_second = (
            pynutil.insert('seconds: "')
            + (
                pynutil.delete("0") + graph_minute_single
                | graph_minute_double
            )
            + pynutil.insert('"')
        )

        final_graph_hour = (
            pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')
        )

        delete_h = pynini.union(
            pynutil.delete(pynini.accep(pynini.escape("h"))),
            pynutil.delete(pynini.accep(pynini.escape("H"))),
        )

        time_delim = pynini.union(
            pynini.accep(pynini.escape(":")),
            pynini.accep(pynini.escape(".")),
        )

        period_rows = load_labels(get_abs_path("data/time/day_period_suffix.tsv"))
        period_branches = []
        for row in period_rows:
            if len(row) < 2 or not row[0].strip():
                continue
            tail, tag_val = row[0].strip(), row[1].strip()
            period_branches.append(
                pynutil.delete(tail) + pynutil.insert(f'suffix: "{tag_val}"')
            )
        suffix_tail = (
            delete_space
            + pynutil.delete("da")
            + delete_space
            + pynini.union(*period_branches)
        )
        optional_suffix = pynini.closure(insert_space + suffix_tail, 0, 1)

        graph_hm = (
            final_graph_hour
            + pynutil.delete(time_delim)
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + optional_suffix
            + pynutil.insert(" preserve_order: true")
        )

        graph_h_minute = (
            final_graph_hour
            + delete_h
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + optional_suffix
            + pynutil.insert(" preserve_order: true")
        )

        graph_h_only = (
            final_graph_hour
            + delete_h
            + optional_suffix
            + pynutil.insert(" preserve_order: true")
        )

        graph_hms = (
            final_graph_hour
            + pynutil.delete(time_delim)
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + pynutil.delete(time_delim)
            + (pynutil.delete("00") | insert_space + final_graph_second)
            + optional_suffix
            + pynutil.insert(" preserve_order: true")
        )

        final_graph = (graph_hm | graph_h_minute | graph_h_only | graph_hms).optimize()
        self.fst = self.add_tokens(final_graph).optimize()
