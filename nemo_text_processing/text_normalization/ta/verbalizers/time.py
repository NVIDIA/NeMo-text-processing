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

HOUR_NOUN, MINUTE_NOUN, SECOND_NOUN = "மணி", "நிமிடம்", "வினாடி"
# A case suffix the tagger carried, and the inflected shape of whichever noun is spoken last.
# Only the last noun takes the case: 10:30க்கு is பத்து மணி முப்பது நிமிடத்திற்கு, not
# பத்து மணிக்கு முப்பது நிமிடத்திற்கு.
CASE_INFLECTIONS = {
    "க்கு": ("மணிக்கு", "நிமிடத்திற்கு", "வினாடிக்கு"),
    "இல்": ("மணியில்", "நிமிடத்தில்", "வினாடியில்"),
}


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "பத்து" minutes: "முப்பது" } -> பத்து மணி முப்பது நிமிடம்
        time { hours: "பத்து" } -> பத்து மணி
        time { hours: "பத்து" minutes: "முப்பது" meridiem: "முற்பகல்" } -> முற்பகல் பத்து மணி முப்பது நிமிடம்
        time { hours: "ஏழு" morphosyntactic_features: "க்கு" } -> ஏழு மணிக்கு

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        def shapes(hour_last: str, minute_last: str, second_last: str) -> 'pynini.FstLike':
            """
            The four spoken shapes. Each noun is spoken plain except the one that ends the
            phrase, which takes the ``_last`` form the caller passes.
            """
            head = hour + insert_space
            graph_h = head + pynutil.insert(hour_last)
            hm_head = head + pynutil.insert(HOUR_NOUN) + delete_space + insert_space + minute + insert_space
            graph_hm = hm_head + pynutil.insert(minute_last)
            graph_hms = (
                hm_head
                + pynutil.insert(MINUTE_NOUN)
                + delete_space
                + insert_space
                + second
                + insert_space
                + pynutil.insert(second_last)
            )
            graph_hs = (
                head
                + pynutil.insert(HOUR_NOUN)
                + delete_space
                + insert_space
                + second
                + insert_space
                + pynutil.insert(second_last)
            )
            return graph_hms | graph_hm | graph_hs | graph_h

        plain = shapes(HOUR_NOUN, MINUTE_NOUN, SECOND_NOUN)
        # A carried case suffix is spoken on the last noun and the field itself is consumed.
        inflected = pynini.union(
            *[
                shapes(*forms) + delete_space + pynutil.delete(f"morphosyntactic_features: \"{written}\"")
                for written, forms in CASE_INFLECTIONS.items()
            ]
        )

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
        self.graph = meridiem + (plain | inflected)
        self.fst = self.delete_tokens(self.graph).optimize()
