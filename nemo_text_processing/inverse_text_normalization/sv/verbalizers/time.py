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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "8" minutes: "30" zone: "e s t" } -> 08:30 est
        time { hours: "8" } -> kl. 8
        time { hours: "8" minutes: "30" seconds: "10" } -> 08:30:10
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        kl_hour = (
            pynutil.delete("hours: \"") + pynini.accep("kl. ") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        )
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        zeroed_hour = hour @ add_leading_zero_to_double_digit
        lead_hour = zeroed_hour | kl_hour
        lead_minute = minute @ add_leading_zero_to_double_digit

        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        lead_second = second @ add_leading_zero_to_double_digit
        ANY_NOT_QUOTE = pynini.closure(NEMO_NOT_QUOTE, 1)
        final_suffix = pynutil.delete("suffix: \"") + ANY_NOT_QUOTE + pynutil.delete("\"")
        optional_suffix = pynini.closure(NEMO_SPACE + final_suffix, 0, 1)
        zone = pynutil.delete("zone: \"") + ANY_NOT_QUOTE + pynutil.delete("\"")
        optional_zone = pynini.closure(NEMO_SPACE + zone, 0, 1)
        one_optional_suffix = NEMO_SPACE + final_suffix + optional_zone
        one_optional_suffix |= optional_suffix + NEMO_SPACE + zone
        graph = (
            delete_space
            + pynutil.insert(":")
            + lead_minute
            + pynini.closure(delete_space + pynutil.insert(":") + lead_second, 0, 1)
            + optional_suffix
            + optional_zone
        )
        graph_h = hour + one_optional_suffix
        graph_klh = kl_hour + optional_suffix + optional_zone
        graph_hm = lead_hour + graph
        final_graph = graph_hm | graph_h | graph_klh
        self.fst = self.delete_tokens(final_graph).optimize()
