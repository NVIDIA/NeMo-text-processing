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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "பத்து" minutes: "முப்பது" } -> பத்து மணி முப்பது நிமிடம்
        time { hours: "பத்து" } -> பத்து மணி
        time { hours: "பத்து" minutes: "முப்பது" meridiem: "முற்பகல்" } -> முற்பகல் பத்து மணி முப்பது நிமிடம்

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        insert_mani = pynutil.insert("மணி")
        insert_minute = pynutil.insert("நிமிடம்")
        insert_second = pynutil.insert("வினாடி")

        graph_h = hour + insert_space + insert_mani
        graph_hm = graph_h + delete_space + insert_space + minute + insert_space + insert_minute
        graph_hms = graph_hm + delete_space + insert_space + second + insert_space + insert_second
        graph_hs = graph_h + delete_space + insert_space + second + insert_space + insert_second

        # A day-part word or a resolved AM/PM is fronted.
        meridiem = pynini.closure(
            pynutil.delete("meridiem: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + insert_space,
            0,
            1,
        )
        self.graph = meridiem + (graph_hms | graph_hm | graph_hs | graph_h)
        self.fst = self.delete_tokens(self.graph).optimize()
