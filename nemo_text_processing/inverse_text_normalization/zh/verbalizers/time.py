# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transcucer for verbalizing time, e.g.,
    time { hours: "12" minutes: "30" } -> 12:30
    time { hours: "1" minutes: "30" } -> 01:30
    time { hours: "1" minutes: "30" affix: "a.m." } -> 01:30 a.m.
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        token_hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete("\"")
        )
        token_minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete("\"")
        )

        affix_am = (
            delete_space
            + pynutil.delete("affix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("a.m.")
            + pynutil.delete("\"")
        )
        affix_am = pynutil.insert(" ") + pynini.closure(affix_am, 0, 1)
        graph_am = token_hour @ add_leading_zero + delete_space + pynutil.insert(":") + token_minute
        graph_am_affix = token_hour @ add_leading_zero + delete_space + pynutil.insert(":") + token_minute + affix_am
        graph_am = graph_am | graph_am_affix

        # 5:00 p.m. -> 17:00 or keep 17:00 as 17:00
        affix_pm = (
            delete_space
            + pynutil.delete("affix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("p.m.")
            + pynutil.delete("\"")
        )
        optional_affix_pm = pynutil.insert(" ") + pynini.closure(affix_pm, 0, 1)
        graph_pm = token_hour @ add_leading_zero + delete_space + pynutil.insert(":") + token_minute
        graph_pm_affix = (
            token_hour @ add_leading_zero
            + delete_space
            + pynutil.insert(":")
            + token_minute
            + pynutil.insert(" ")
            + affix_pm
        )
        graph_pm = graph_pm | graph_pm_affix

        final_graph = graph_am | graph_pm
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
