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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import GraphFst, insert_space
from nemo_text_processing.inverse_text_normalization.ja.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifing time, e.g.,
    一時三十分 -> time { hours: "1" minutes: "0" }
    五時二十分過ぎ -> time { hours: "5" minutes: "20" suffix: "過ぎ" }
    八時半頃 -> time { hours: "8" minutes: "半" suffix: "頃" }
    十時五分前 -> time { hours: "10" minutes: "25" suffix: "前" }
    正午一分前 -> time { hours: "正午" minutes: "1" suffix: "前" }
    正午十分過ぎ -> time { hours: "正午" minutes: "10" suffix: "過ぎ" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        hours = pynini.string_file(get_abs_path("data/time_hours.tsv"))
        minutes_seconds = pynini.string_file(get_abs_path("data/time_minutes_seconds.tsv"))

        hour_component = (
            pynutil.insert("hours: \"") + ((hours + pynutil.delete("時")) | pynini.accep("正午")) + pynutil.insert("\"")
        )
        minute_component = (
            pynutil.insert("minutes: \"")
            + ((minutes_seconds + pynutil.delete("分")) | pynini.accep("半"))
            + pynutil.insert("\"")
        )
        second_component = pynutil.insert("seconds: \"") + minutes_seconds + pynutil.delete("秒") + pynutil.insert("\"")

        graph_regular = (
            pynini.closure(hour_component + insert_space + minute_component + insert_space + second_component)
            | pynini.closure(hour_component | minute_component | second_component)
            | pynini.closure(hour_component + insert_space + minute_component)
            | pynini.closure(minute_component + insert_space + second_component)
        )

        words = pynini.accep("前") | pynini.accep("過ぎ") | pynini.accep("頃")
        suffix = pynutil.insert("suffix: \"") + words + pynutil.insert("\"")
        graph = graph_regular + pynini.closure(insert_space + suffix)

        final_graph = graph

        final_graph = self.add_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
