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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing times, e.g.
        time { hours: "10" minutes: "30" preserve_order: true } -> 10:30
        time { hours: "10" preserve_order: true } -> 10:00
        time { morphosyntactic_features: "காலை" hours: "10" preserve_order: true } -> காலை 10:00
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        two_digits = pynini.union(NEMO_DIGIT + NEMO_DIGIT, pynutil.insert("0") + NEMO_DIGIT).optimize()
        hours = pynutil.delete("hours: \"") + pynini.closure(NEMO_DIGIT, 1, 2) + pynutil.delete("\"")
        minutes = pynutil.delete("minutes: \"") + two_digits + pynutil.delete("\"")
        seconds = pynutil.delete("seconds: \"") + two_digits + pynutil.delete("\"")
        # A fronted day-part word is written before the time.
        day_part = pynini.closure(
            pynutil.delete("morphosyntactic_features: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + insert_space,
            0,
            1,
        )

        graph_h = hours + pynutil.insert(":00")
        graph_hm = hours + delete_space + pynutil.insert(":") + minutes
        graph_hms = graph_hm + delete_space + pynutil.insert(":") + seconds
        graph_hs = hours + pynutil.insert(":00:") + delete_space + seconds
        self.graph = day_part + (graph_hms | graph_hm | graph_hs | graph_h) + delete_preserve_order
        self.fst = self.delete_tokens(self.graph).optimize()
