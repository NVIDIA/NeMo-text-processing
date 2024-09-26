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

from nemo_text_processing.inverse_text_normalization.mr.graph_utils import GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.mr.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. साडे चार -> time { hours: "४" minutes: "३०" }
        e.g. सव्वा बारा -> time { hours: "१२" minutes: "१५" }
        e.g. पावणे दहा -> time { hours: "९" minutes: "४५" }
        e.g. अकराला पाच मिनिटे -> time { hours: "१०" minutes: "५५" }
        e.g. अकरा वाजून दोन मिनिटे -> time { hours: "११" minutes: "२" }
        e.g. अडीच -> time { hours: "२" minutes: "३०" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        hours = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        minutes = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
        hours_to = pynini.string_file(get_abs_path("data/time/hours_to.tsv"))
        minutes_to = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))

        time_word = pynini.cross("वाजून", "")
        minutes_word = pynini.cross("मिनिटे", "") | pynini.cross("मिनिट", "")
        graph_time_full = (
            pynutil.insert("hours: \"")
            + hours
            + pynutil.insert("\"")
            + delete_space
            + time_word
            + delete_space
            + pynutil.insert(" ")
            + pynutil.insert("minutes: \"")
            + minutes
            + pynutil.insert("\"")
            + delete_space
            + minutes_word
        )
        graph_time_to = (
            pynutil.insert("hours: \"")
            + hours_to
            + pynutil.insert("\"")
            + pynini.cross("ला", "")
            + delete_space
            + pynutil.insert(" ")
            + pynutil.insert("minutes: \"")
            + minutes_to
            + pynutil.insert("\"")
            + delete_space
            + minutes_word
        )

        # special terms used for 15, 30 and 45 minutes
        graph_fifteen = (
            pynini.cross("सव्वा", "")
            + delete_space
            + pynutil.insert("hours: \"")
            + hours
            + pynutil.insert("\"")
            + pynutil.insert(" ")
            + pynutil.insert("minutes: \"")
            + pynutil.insert("१५")
            + pynutil.insert("\"")
        )
        graph_thirty = (
            pynini.cross("साडे", "")
            + delete_space
            + pynutil.insert("hours: \"")
            + hours
            + pynutil.insert("\"")
            + pynutil.insert(" ")
            + pynutil.insert("minutes: \"")
            + pynutil.insert("३०")
            + pynutil.insert("\"")
        )
        graph_fortyfive = (
            pynini.cross("पावणे", "")
            + delete_space
            + pynutil.insert("hours: \"")
            + hours_to
            + pynutil.insert("\"")
            + pynutil.insert(" ")
            + pynutil.insert("minutes: \"")
            + pynutil.insert("४५")
            + pynutil.insert("\"")
        )

        special_cases = (pynini.cross("दीड", "") + pynutil.insert("hours: \"१\" minutes: \"३०\"")) | (
            pynini.cross("अडीच", "") + pynutil.insert("hours: \"२\" minutes: \"३०\"")
        )

        graph = pynini.union(
            graph_time_full, graph_time_to, graph_fifteen, graph_thirty, graph_fortyfive, special_cases
        )

        final_graph = graph
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
