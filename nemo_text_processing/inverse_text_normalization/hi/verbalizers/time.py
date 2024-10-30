# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_HI_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        e.g. एक बजके सात मिनट -> time { hours: "१" minutes: "७" }
        e.g. चार बजे चवालीस मिनट -> time { hours: "४" minutes: "४४" }
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_HI_DIGIT, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_HI_DIGIT, 1)
            + pynutil.delete("\"")
        )
        second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_HI_DIGIT, 1)
            + pynutil.delete("\"")
        )

        graph_hour = hour + delete_space + pynutil.insert(":") + delete_space + pynutil.insert("००") + delete_space

        # hour minute second
        graph_hms = (
            hour
            + delete_space
            + pynutil.insert(":")
            + delete_space
            + minute
            + delete_space
            + pynutil.insert(":")
            + delete_space
            + second
            + delete_space
        )

        # hour minute
        graph_hm = hour + delete_space + pynutil.insert(":") + delete_space + minute + delete_space

        # hour second
        graph_hs = (
            hour
            + delete_space
            + pynutil.insert(":")
            + delete_space
            + pynutil.insert("००")
            + delete_space
            + pynutil.insert(":")
            + second
            + delete_space
        )

        # minute second
        graph_ms = (
            pynutil.insert("००")
            + delete_space
            + pynutil.insert(":")
            + delete_space
            + minute
            + delete_space
            + pynutil.insert(":")
            + delete_space
            + second
            + delete_space
        )

        graph = graph_hour | graph_hms | graph_hm | graph_hs | graph_ms

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
