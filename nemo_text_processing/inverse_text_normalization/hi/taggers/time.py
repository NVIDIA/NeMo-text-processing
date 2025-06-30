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
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
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

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        hour_graph = pynini.string_file(get_abs_path("data/time/hour.tsv")).invert()
        cardinal_graph = cardinal.graph_single_digit_with_zero | cardinal.graph_teens_and_ties
        paune_hour_graph = pynini.string_file(get_abs_path("data/time/hour_for_paune.tsv")).invert()

        delete_baje = pynini.union(
            pynutil.delete("बजके") | pynutil.delete("बजकर") | pynutil.delete("बजे") | pynutil.delete("घंटा")
        )

        delete_minute = pynutil.delete("मिनट")
        delete_second = pynutil.delete("सेकंड")

        self.hour = pynutil.insert("hours: \"") + hour_graph + pynutil.insert("\" ")
        self.paune_hour = pynutil.insert("hours: \"") + paune_hour_graph + pynutil.insert("\" ")
        self.minute = pynutil.insert("minutes: \"") + cardinal_graph + pynutil.insert("\" ")
        self.second = pynutil.insert("seconds: \"") + cardinal_graph + pynutil.insert("\" ")

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
        graph_hm = pynutil.add_weight(
            self.hour
            + delete_space
            + pynini.closure(delete_baje, 0, 1)
            + delete_space
            + self.minute
            + pynini.closure(delete_space + delete_minute, 0, 1),
            0.01,
        )

        # hour second
        graph_hs = pynutil.add_weight(
            self.hour + delete_space + delete_baje + delete_space + self.second + delete_space + delete_second, 0.01
        )

        # minute second
        graph_ms = (
            self.minute + delete_space + delete_minute + delete_space + self.second + delete_space + delete_second
        )

        # hour
        graph_hour = self.hour + delete_space + delete_baje

        graph_saade = pynutil.add_weight(
            pynutil.delete("साढ़े")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"३०\"")
            + delete_space
            + pynini.closure(delete_baje),
            0.01,
        )
        graph_sava = pynutil.add_weight(
            pynutil.delete("सवा")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"१५\"")
            + delete_space
            + pynini.closure(delete_baje),
            0.01,
        )
        graph_paune = pynutil.add_weight(
            pynutil.delete("पौने")
            + delete_space
            + self.paune_hour
            + delete_space
            + pynutil.insert(" minutes: \"४५\"")
            + delete_space
            + pynini.closure(delete_baje),
            0.01,
        )
        graph_dedh = pynutil.add_weight(
            pynini.union(pynutil.delete("डेढ़") | pynutil.delete("डेढ़"))
            + delete_space
            + delete_baje
            + pynutil.insert("hours: \"१\"")
            + delete_space
            + pynutil.insert(" minutes: \"३०\""),
            0.01,
        )
        graph_dhaai = pynutil.add_weight(
            pynutil.delete("ढाई")
            + delete_space
            + delete_baje
            + pynutil.insert("hours: \"२\"")
            + delete_space
            + pynutil.insert(" minutes: \"३०\""),
            0.01,
        )
        graph_quarterly_measures = (
            graph_dedh
            | graph_dhaai
            | ((graph_saade | graph_sava | graph_paune) + pynini.closure(delete_space + delete_baje))
        )

        graph = graph_hms | graph_hm | graph_hs | graph_ms | graph_hour | graph_quarterly_measures
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
