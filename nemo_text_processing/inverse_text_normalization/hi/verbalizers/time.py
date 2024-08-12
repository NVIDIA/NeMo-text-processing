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
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_HI_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "12" minutes: "30" } -> 12:30
        time { hours: "1" minutes: "12" } -> 01:12
        time { hours: "2" suffix: "a.m." } -> 02:00 a.m.
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (NEMO_HI_DIGIT + NEMO_HI_DIGIT) | (pynutil.insert("0") + NEMO_HI_DIGIT)
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
        suffix = (
            delete_space
            + insert_space
            + pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        #deleted optional suffix as not applicable for hindi (zone)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()