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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


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
        super().__init__(name="time", kind="verbalize")

        hours_component = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        minutes_component = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        seconds_component = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        suffix_component = pynutil.delete("suffix: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        prefix_component = pynutil.delete("prefix: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        # Add a leading zero to single-digit minutes/seconds
        single_digit = NEMO_DIGIT
        leading_zero = pynutil.insert("0") + single_digit
        add_leading_zero = pynini.union(single_digit @ leading_zero, pynini.closure(NEMO_DIGIT, 2))

        minutes = minutes_component @ add_leading_zero
        seconds = seconds_component @ add_leading_zero

        # Defining all the possible combinations
        path_h = hours_component + pynutil.insert(":00")
        path_m = minutes
        path_s = seconds

        path_hm = hours_component + delete_space + pynutil.insert(":") + minutes
        path_hs = (
            hours_component
            + delete_space
            + pynutil.insert(":")
            + pynutil.insert("00")
            + delete_space
            + pynutil.insert(":")
            + seconds
        )
        path_ms = minutes + delete_space + pynutil.insert(":") + seconds

        path_hms = (
            hours_component
            + delete_space
            + pynutil.insert(":")
            + minutes
            + delete_space
            + pynutil.insert(":")
            + seconds
        )

        time_graph = pynini.union(path_h, path_m, path_s, path_hm, path_hs, path_ms, path_hms)

        # Adding prefix and suffix space
        optional_prefix_out = pynini.closure(delete_space + prefix_component, 0, 1)
        optional_suffix_out = pynini.closure(delete_space + pynutil.insert(" ") + suffix_component, 0, 1)

        final_graph = optional_prefix_out + time_graph + optional_suffix_out
        self.fst = self.delete_tokens(delete_space + final_graph).optimize()
