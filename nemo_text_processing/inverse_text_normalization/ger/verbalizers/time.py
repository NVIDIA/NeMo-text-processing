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
from nemo_text_processing.inverse_text_normalization.ger.data.time.time_frame_mappings import (
    time_periods_and_mappings,
)
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    NEMO_ALPHA,
    NEMO_SIGMA,
    GraphFst,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time,
        e.g. time { hours: "8" minutes: "30" zone: "e s t" } -> 8:30 Uhr est
        e.g. time { hours: "8" } -> 8:00 Uhr
        e.g. time { hours: "8" minutes: "30" seconds: "10" } -> 8:30:10 Uhr
        e.g. time { suffix: "ab" hours: "3" minutes: "25" suffix: "nachmittags"} -> ab 15:25 Uhr
        e.g. time { suffix: "zwischen" hours: "3" minutes: "25" suffix: "nachmittags" suffix: "und" hours: "5" minutes: "30" suffix: "nachmittags" } -> zwischen 15:25 Uhr und 17:30 Uhr
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        # Implements base components of the graph: hh:mm:ss

        graph_hours_solo = (
            pynutil.delete("hours:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
            + pynutil.insert(":00")
        )

        graph_hours_component = (
            pynutil.delete("hours:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )

        graph_minutes = (
            pynutil.delete("minutes:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )

        graph_seconds = (
            pynutil.delete("seconds:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )

        # Combines the base components

        graph_hms = (
            graph_hours_component
            + pynini.cross(NEMO_SPACE, ":")
            + graph_minutes
            + pynini.cross(NEMO_SPACE, ":")
            + graph_seconds
        )
        graph_hm = graph_hours_component + pynini.cross(NEMO_SPACE, ":") + graph_minutes
        graph_hs = (
            graph_hours_component
            + pynini.cross(NEMO_SPACE, ":")
            + pynutil.insert("00")
            + pynutil.insert(":")
            + graph_seconds
        )

        graph_base_time = graph_hours_solo | graph_hms | graph_hm | graph_hs

        # Implements the logic for time zones

        graph_timezone = (
            pynutil.delete("zone:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + (
                pynini.closure(NEMO_ALPHA, 3, 3)
                | pynini.closure(NEMO_ALPHA, 3, 3)
                + (pynini.accep("+") | pynini.accep("-"))
                + NEMO_DIGIT
                + pynini.accep(".5").ques
            )
            + pynutil.delete('"')
        )

        # Implements the logic for colloquial to military time transduction based on the six basic time frames in the day.
        # Each time frame utilizes different transduction rules (e.g. ein Uhr nachmittags -> 13:00, but ein Uhr nachts -> 1:00)

        time_frames_and_mappings = time_periods_and_mappings
        am_pm_conversion_graphs = []

        for time, mapping in time_frames_and_mappings.items():
            graph_period = (
                pynutil.delete("suffix:")
                + pynutil.delete(NEMO_SPACE)
                + pynutil.delete('"')
                + pynutil.delete(time)
                + pynutil.delete('"')
            )
            map = pynini.string_map(mapping)
            graph_map = pynini.cdrewrite(map, "[BOS]", ":", NEMO_SIGMA)

            graph_conversion = (graph_base_time @ (graph_map)) + (
                pynutil.delete(NEMO_SPACE) + graph_period
            )

            am_pm_conversion_graphs.append(graph_conversion)

        # The postposition "morgens" requires a simple acceptor

        graph_morgens = (
            pynutil.delete("suffix:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynutil.delete("morgens")
            + pynutil.delete('"')
        )

        graph_morgens = graph_base_time + pynutil.delete(NEMO_SPACE) + graph_morgens

        graph_am_pm = pynini.union(*am_pm_conversion_graphs) | graph_morgens

        insert_Uhr = pynutil.insert(NEMO_SPACE) + pynutil.insert("Uhr")

        graph_time = (
            (graph_base_time | graph_am_pm)
            + insert_Uhr
            + (pynini.accep(NEMO_SPACE) + graph_timezone).ques
        )

        # Handles prepositional time expressions

        prep_and_conj = pynini.string_file(get_abs_path("data/time/prep_conj.tsv"))
        graph_prep_or_conj = (
            pynutil.delete("morphosyntactic_features:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + prep_and_conj
            + pynutil.delete('"')
        )

        # Combines everything

        graph_time_final = (
            graph_prep_or_conj + pynini.accep(NEMO_SPACE)
        ).ques + graph_time
        delete_tokens = self.delete_tokens(graph_time_final)
        self.fst = delete_tokens.optimize()
