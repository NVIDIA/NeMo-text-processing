# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.mr.graph_utils import NEMO_DIGIT, GraphFst, delete_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        e.g. time { hours: "४" minutes: "३०" } -> ०४:३०
        e.g. time { hours: "११" minutes: "३०" } -> ११:३०
        e.g. time { hours: "८" minutes: "१५" } -> ०८:१५
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("०") + NEMO_DIGIT)
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        graph = (
            (hour @ add_leading_zero_to_double_digit)
            + delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
