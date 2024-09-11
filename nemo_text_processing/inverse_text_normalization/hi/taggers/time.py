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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class TimeFst(GraphFst):
    """
        Finite state transducer for classifying time, 
        e.g. एक बजके सात मिनट -> time { hours: "१" minutes: "७" }
        e.g. चार बजे चवालीस मिनट -> time { hours: "४" minutes: "४४" }    
    Args:
        cardinal: CardinalFst
        time: TimeFst
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        hour_graph = pynini.string_file(get_abs_path("data/time/hour.tsv")).invert()
        minute_graph = pynini.string_file(get_abs_path("data/time/minute_and_second.tsv")).invert()
        second_graph = pynini.string_file(get_abs_path("data/time/minute_and_second.tsv")).invert()

        delete_baje = pynini.union(
            pynutil.delete("बजके") | pynutil.delete("बजकर") | pynutil.delete("बजे") | pynutil.delete("घंटा")
        )

        delete_minute = pynutil.delete("मिनट")
        delete_second = pynutil.delete("सेकंड")

        self.hour = pynutil.insert("hours: \"") + hour_graph + pynutil.insert("\" ")
        self.minute = pynutil.insert("minutes: \"") + minute_graph + pynutil.insert("\" ")
        self.second = pynutil.insert("seconds: \"") + second_graph + pynutil.insert("\" ")

        # hour minute second
        graph_hms = (
            self.hour
            + delete_space
            + delete_baje
            + delete_space
            + self.minute
            + delete_space
            + delete_minute
            + delete_space
            + self.second
            + delete_space
            + delete_second
        )

        # hour minute and hour minute without "baje and minat"
        graph_hm = (
            self.hour
            + delete_space
            + pynini.closure(delete_baje, 0, 1)
            + delete_space
            + self.minute
            + pynini.closure(delete_space + delete_minute, 0, 1)
        )

        # hour second
        graph_hs = self.hour + delete_space + delete_baje + delete_space + self.second + delete_space + delete_second

        # minute second
        graph_ms = (
            self.minute + delete_space + delete_minute + delete_space + self.second + delete_space + delete_second
        )

        # hour
        graph_hour = self.hour + delete_space + delete_baje

        graph = graph_hms | graph_hm | graph_hs | graph_ms | graph_hour
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
