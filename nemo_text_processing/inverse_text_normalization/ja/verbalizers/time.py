# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.,
    time { hours: "1" minutes: "0" } -> 1時30分 ->
    time { hours: "5" minutes: "20" suffix: "過ぎ" } -> 5時20分
    time { hours: "8" minutes: "半" suffix: "頃" } -> 8時半頃
    time { hours: "10" minutes: "25" suffix: "前" } -> 10時5分前
    time { hours: "正午" minutes: "1" suffix: "前" } -> 正午1分前
    time { hours: "正午" minutes: "10" suffix: "過ぎ" } -> 正午10分過ぎ
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hours_component = (
            pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("時") + pynutil.delete("\"")
        )
        hours_component_alt = pynutil.delete("hours: \"") + pynini.accep("正午") + pynutil.delete("\"")
        hours_component |= hours_component_alt

        minutes_component = (
            pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("分") + pynutil.delete("\"")
        )
        minutes_component_alt = pynutil.delete("minutes: \"") + pynini.accep("半") + pynutil.delete("\"")
        minutes_component |= minutes_component_alt
        second_component = (
            pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.insert("秒") + pynutil.delete("\"")
        )
        suffix_component = pynutil.delete("suffix: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_regular_time = (
            pynini.closure(hours_component)
            + pynini.closure(delete_space + minutes_component)
            + pynini.closure(delete_space + second_component)
            + pynini.closure(delete_space + suffix_component)
        )
        graph = graph_regular_time

        final_graph = graph

        final_graph = self.delete_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
