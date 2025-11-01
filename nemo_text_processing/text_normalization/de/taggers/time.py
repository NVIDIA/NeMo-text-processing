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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, convert_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15 Uhr est" -> time { hours: "2" minutes: "15" zone: "e s t"}
        "2 Uhr" -> time { hours: "2" }
        "09:00 Uhr" -> time { hours: "2" }
        "02:15:10 Uhr" -> time { hours: "2" minutes: "15" seconds: "10"}
        "04:30" -> time { hours: "4" minutes: "30"}

    durations (with explicit duration cue)
        "Der kenianische Athlet stellte mit 2:00:35 eine Weltrekordzeit bei den Männern auf."
        -> ... time { hours: "2" minutes: "0" seconds: "35" mode: "duration" preserve_order: true }
        "Für Frauen wäre eine Zeit unter 04:30 ebenfalls sehr gut."
        -> ... time { minutes: "4" seconds: "30" mode: "duration" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based / ITN normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        final_suffix = pynutil.delete(" ") + pynutil.delete("Uhr") | pynutil.delete("uhr")
        time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))

        labels_hour = [str(x) for x in range(0, 25)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (pynutil.delete("0").ques | (NEMO_DIGIT - "0")) + NEMO_DIGIT

        graph_hour = pynini.union(*labels_hour)

        graph_minute_single = pynini.union(*labels_minute_single)
        graph_minute_double = pynini.union(*labels_minute_double)

        final_graph_hour_only = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')
        final_graph_hour = (
            pynutil.insert('hours: "') + delete_leading_zero_to_double_digit @ graph_hour + pynutil.insert('"')
        )

        final_graph_minute = (
            pynutil.insert('minutes: "')
            + (pynutil.delete("0").ques + (graph_minute_single | graph_minute_double))
            + pynutil.insert('"')
        )
        final_graph_second = (
            pynutil.insert('seconds: "')
            + (pynutil.delete("0").ques + (graph_minute_single | graph_minute_double))
            + pynutil.insert('"')
        )
        final_time_zone_optional = pynini.closure(
            pynini.accep(" ") + pynutil.insert('zone: "') + convert_space(time_zone_graph) + pynutil.insert('"'), 0, 1
        )

        # accepts explicit 'Uhr' format: 02:30 Uhr, 02.30 Uhr, 2:30 Uhr, 2.30 Uhr
        graph_hm = (
            final_graph_hour
            + (pynutil.delete(":") | pynutil.delete("."))
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + final_suffix
            + pynutil.insert(' has_uhr: "true"')
            + final_time_zone_optional
        )
        # accepts bare HH:MM format (e.g. '04:30') - no dot delimiter to avoid colliding with date formats (eg. '02.03')
        graph_hm_no_uhr = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + final_time_zone_optional
        )

        # accepts explicit 'Uhr' format: 10:30:05 Uhr
        graph_hms = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", ' minutes: "0"') | (insert_space + final_graph_minute))
            + pynutil.delete(":")
            + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second))
            + final_suffix
            + pynutil.insert(' has_uhr: "true"')
            + final_time_zone_optional
            + pynutil.insert(" preserve_order: true")
        )
        # graph_hms for bare HH:MM:SS without 'Uhr' (e.g. '04:30:15')
        graph_hms_no_uhr = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", ' minutes: "0"') | (insert_space + final_graph_minute))
            + pynutil.delete(":")
            + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second))
            + final_time_zone_optional
            + pynutil.insert(' preserve_order: true')
        )

        # 2 Uhr est (explicit Uhr)
        graph_h = final_graph_hour_only + final_suffix + pynutil.insert(' has_uhr: "true"') + final_time_zone_optional

        # Special-case: preposition-led duration cues (e.g., 'unter 04:30', 'mit 2:00:35')
        duration_cues = pynini.string_file(get_abs_path("data/time/duration_cues.tsv"))
        preposition_prefix = (
            pynutil.insert(' preposition: "') + duration_cues + pynutil.delete(" ") + pynutil.insert('" ')
        )

        # Preposition-led MMSS (e.g. 'unter 04:30')
        graph_ms_dur = (
            preposition_prefix
            + final_graph_minute
            + pynutil.delete(":")
            + (pynutil.delete("00") | (insert_space + final_graph_second))
            + final_time_zone_optional
        )

        # Preposition-led HHMMSS (e.g. 'mit 2:00:35')
        graph_hms_dur = (
            preposition_prefix
            + final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", ' minutes: "0"') | (insert_space + final_graph_minute))
            + pynutil.delete(":")
            + (pynini.cross("00", ' seconds: "0"') | (insert_space + final_graph_second))
            + final_time_zone_optional
            + pynutil.insert(" preserve_order: true")
        )

        final_graph = (
            graph_hm | graph_h | graph_hms | graph_ms_dur | graph_hms_dur | graph_hms_no_uhr | graph_hm_no_uhr
        ).optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
