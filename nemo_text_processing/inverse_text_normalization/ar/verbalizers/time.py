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

from nemo_text_processing.text_normalization.ar.graph_utils import NEMO_DIGIT, NEMO_NOT_QUOTE, GraphFst, delete_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing a time token into digits, e.g.
        time { hours: "8" minutes: "30" } -> 08:30
        time { hours: "8" minutes: "30" seconds: "10" } -> 08:30:10
        time { hours: "9" suffix: "صباحًا" } -> 9 صباحًا
        time { hours: "8" } -> 8
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        zone = pynutil.delete("zone: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)
        suffix = pynutil.delete("suffix: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_suffix = pynini.closure(pynini.accep(" ") + suffix, 0, 1)

        graph_minutes_seconds = (
            delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
            + pynini.closure(delete_space + pynutil.insert(":") + (second @ add_leading_zero_to_double_digit), 0, 1)
        )
        graph_h = hour
        graph_hms = hour @ add_leading_zero_to_double_digit + graph_minutes_seconds
        final_graph = (graph_hms | graph_h) + optional_suffix + optional_zone
        self.fst = self.delete_tokens(final_graph).optimize()
