# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_SPACE, GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. 열두시 삼십분 -> time { hours: "12" minutes: "30" }
        e.g. 12분전 -> time { minutes: "12" suffix: "전" }
        e.g. 새벽 두시 -> time { hours: "2" suffix: "새벽" }
        e.g. 두시반 -> time { hours: "2" minutes: "30" }
        e.g. 오후 두시반 -> time { prefix: "오후" hours: "2" minutes: "30" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        # 1-9 in cardinals for minutes and seconds
        cardinal_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        cardinal_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        graph_tens_prefix = pynini.string_file(get_abs_path("data/time/ten_prefix.tsv"))

        # Graphing 10-19
        graph_ten = pynini.union(pynini.cross("십", "10"), pynini.cross("십", "1") + cardinal_digit).optimize()
        # Graphing 20-59
        graph_tens = (graph_tens_prefix + pynini.cross("십", "0")) | (
            graph_tens_prefix + pynini.cross("십", "") + cardinal_digit
        )

        graph_0_to_59 = pynini.union(cardinal_zero, cardinal_digit, graph_ten, graph_tens).optimize()

        # 1-12 for hours
        graph_hours = pynini.string_file(get_abs_path("data/time/time_hours.tsv"))

        # Adding space if there are one
        spacing = pynini.closure(pynini.accep(NEMO_SPACE), 0, 1)

        hour_suffix = pynini.cross("시", "")
        minute_suffix = pynini.cross("분", "")
        second_suffix = pynini.cross("초", "")

        hour_component = pynutil.insert("hours: \"") + (graph_hours + spacing + hour_suffix) + pynutil.insert("\"")

        # half minute only allowed after hours: "두시반" / "두시 반"
        half_minute_component = pynutil.insert('minutes: "30"') + spacing + pynini.cross("반", "")

        minute_component = (
            pynutil.insert("minutes: \"") + (graph_0_to_59 + spacing + minute_suffix) + pynutil.insert("\"")
        )

        second_component = (
            pynutil.insert("seconds: \"") + (graph_0_to_59 + spacing + second_suffix) + pynutil.insert("\"")
        )

        hm_opt = pynini.closure(delete_space + minute_component, 0, 1)
        hs_opt = pynini.closure(delete_space + second_component, 0, 1)

        h_half = hour_component + delete_space + half_minute_component + hs_opt
        hms = hour_component + hm_opt + hs_opt
        ms = minute_component + pynini.closure(delete_space + second_component, 0, 1)
        s_only = second_component

        graph_regular = pynini.union(h_half, hms, ms, s_only).optimize()

        # 오전 = AM, 오후 = PM
        ampm_words = pynini.union("오전", "오후", "새벽", "아침")
        ampm_tag = pynutil.insert('suffix: "') + ampm_words + pynutil.insert('"')

        # 전 = before, 후 = after
        suffix_words = pynini.accep("전") | pynini.accep("후")
        suffix_tag = pynutil.insert("suffix: \"") + suffix_words + pynutil.insert("\"")

        time_graph = (
            pynini.closure(delete_space + ampm_tag, 0, 1)
            + graph_regular
            + pynini.closure(delete_space + suffix_tag, 0, 1)
        )

        # Adding cardinal graph to prevent processing out of range numbers
        final_graph = time_graph

        self.fst = self.add_tokens(final_graph).optimize()
