# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import (
    GraphFst,
    insert_space,
    delete_space,
)
from nemo_text_processing.text_normalization.ko.utils import get_abs_path

class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        오전 10시 30분  -> time { suffix: "오전" hours: "열시" minutes: "삼십분" }
        오후 3시 반     -> time { suffix: "오후" hours: "세시" minutes: "삼십분" }
        자정            -> time { hours: "영시" }
        정오            -> time { hours: "열두시" }

    Args:
        cardinal: CardinalFst (Korean cardinal graph)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        # Base number-to-words graph from the Cardinal Fst
        graph_cardinal = cardinal.graph
        strip0 = pynini.closure(pynutil.delete("0"), 0, 1)

        SP = pynini.closure(delete_space)
        SEP = SP + insert_space
        hour_clock = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        division = pynini.string_file(get_abs_path("data/time/division.tsv"))
        
        division_component = pynutil.insert("suffix: \"") + division + pynutil.insert("\"")
        
        # --- Special single-word times ---
        noon = pynini.accep("정오")
        midnight = pynini.accep("자정")
        noon_component = pynutil.insert("hours: \"") + pynini.cross(noon, "열두시") + pynutil.insert("\"")
        midnight_component = pynutil.insert("hours: \"") + pynini.cross(midnight, "영시") + pynutil.insert("\"")

        # --- Korean Hangul components (H시 [M분|반] [S초]) ---
        # "0" or "00" -> 0
        h_zero = strip0 + pynini.accep("0")
        # "13".."24"
        h_13_24 = pynini.union(*[str(i) for i in range(13, 25)])

        # "0시" -> "영시"
        hour_component_ko_zero = (
            pynutil.insert("hours: \"")
            + pynini.cross(h_zero, "영")
            + pynutil.delete("시")
            + pynutil.insert("시")
            + pynutil.insert("\"")
        )

        # "13시..24시" -> Sino-Korean words (십삼/…/이십사) + 시
        hour_component_ko_13_24 = (
            pynutil.insert("hours: \"")
            + (h_13_24 @ graph_cardinal)
            + pynutil.delete("시")
            + pynutil.insert("시")
            + pynutil.insert("\"")
        )

        # "1시..12시" -> Native Korean words (한/두/세/네/…/열두) + 시
        hour_component_ko_1_12 = (
            pynutil.insert("hours: \"")
            + (strip0 + hour_clock)
            + pynutil.delete("시")
            + pynutil.insert("시")
            + pynutil.insert("\"")
        )

        # Priority: 13-24 > 0 > 1-12 to prevent partial matching errors
        hour_component_ko = (
            hour_component_ko_13_24
            | hour_component_ko_zero
            | hour_component_ko_1_12
        ).optimize()
        
        # Minutes: number+"분" or "반" (approx. 30분). Allows optional '쯤|경' after minutes/반.
        about_word = pynini.union("쯤", "경")
        minute_number = (
            pynutil.insert("minutes: \"")
            + (strip0 + graph_cardinal)
            + pynutil.delete("분")
            + pynutil.insert("분")
            + pynutil.insert("\"")
        )
        minute_half = (
            pynutil.insert("minutes: \"")
            + pynutil.delete("반")
            + pynutil.insert("반")
            + pynini.closure(about_word, 0, 1)
            + pynutil.insert("\"")
        )
        minute_component_ko = (minute_half | minute_number).optimize()

        second_component_ko = (
            pynutil.insert("seconds: \"")
            + (strip0 + graph_cardinal)
            + pynutil.delete("초")
            + pynutil.insert("초")
            + pynutil.insert("\"")
        )

        # Allow suffix before or after
        suffix_prefix_opt = pynini.closure(division_component + SEP, 0, 1)
        suffix_postfix_opt = pynini.closure(SEP + division_component, 0, 1)

        # Hangul patterns
        graph_hangul = (
            suffix_prefix_opt
            + (
                hour_component_ko
                | (hour_component_ko + SEP + minute_component_ko)
                | (hour_component_ko + SEP + minute_component_ko + SEP + second_component_ko)
                | minute_component_ko
                | (minute_component_ko + SEP + second_component_ko)
                | second_component_ko
            )
            + suffix_postfix_opt
        ).optimize()

        # Special words with optional suffix
        graph_special = (
            suffix_prefix_opt
            + (noon_component | midnight_component)
            + suffix_postfix_opt
        ).optimize()

        # --- Clock patterns: HH:MM[:SS] ---
        colon = pynutil.delete(":")
        
        # Map 1-12 hours using native-Korean words, allowing an optional leading zero.
        graph_hour_1_12 = (
            pynutil.insert("hours: \"")
            + (strip0 + hour_clock)
            + pynutil.insert("시")
            + pynutil.insert("\"")
        ).optimize()

        # 0, 00, and 13-24 -> Sino-Korean words
        hour_sino_val = (
            pynini.cross("00", "0")
            | pynini.cross("0", "0")
            | pynini.union(*[pynini.cross(str(i), str(i)) for i in range(13, 25)])
        )
        hour_sino_read = hour_sino_val @ graph_cardinal

        graph_hour_others = (
            pynutil.insert("hours: \"")
            + hour_sino_read
            + pynutil.insert("시")
            + pynutil.insert("\"")
        )

        hour_clock_component = (graph_hour_1_12 | graph_hour_others).optimize()

        minute_clock_component = (
            pynutil.insert("minutes: \"")
            + strip0
            + graph_cardinal
            + pynutil.insert("분")
            + pynutil.insert("\"")
        )
        second_clock_component = (
            pynutil.insert("seconds: \"")
            + strip0
            + graph_cardinal
            + pynutil.insert("초")
            + pynutil.insert("\"")
        )

        # HH:MM (drop minutes if "00")
        graph_hm_clock = (
            suffix_prefix_opt
            + hour_clock_component
            + delete_space.ques
            + colon
            + delete_space.ques
            + (pynini.cross("00", "") | pynini.closure(insert_space + minute_clock_component, 0, 1))
            + suffix_postfix_opt
        ).optimize()

        # HH:MM:SS (drop minutes/seconds if "00")
        graph_hms_clock = (
            suffix_prefix_opt
            + hour_clock_component
            + delete_space.ques
            + colon
            + delete_space.ques
            + (pynini.cross("00", "") | pynini.closure(insert_space + minute_clock_component, 0, 1)) 
            + delete_space.ques
            + colon
            + delete_space.ques
            + (pynini.cross("00", "") | pynini.closure(insert_space + second_clock_component, 0, 1))
            + suffix_postfix_opt
        ).optimize()

        graph = (graph_special | graph_hangul | graph_hm_clock | graph_hms_clock).optimize()
        graph_final = self.add_tokens(graph)
        self.fst = graph_final.optimize()
