# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "बारह"  minutes: "दस"  seconds: "दस" } -> बारह बजकर दस मिनट दस सेकंड
        time { hours: "सात" minutes: "चालीस"" } -> सात बजकर चालीस मिनट
        time { hours: "दस" } -> दस बजे

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + insert_space

        minute = (
            pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + insert_space
        )

        second = (
            pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + insert_space
        )

        insert_minute = pynutil.insert("मिनट")
        insert_second = pynutil.insert("सेकंड")
        insert_bajkar = pynutil.insert("बजकर")
        insert_baje = pynutil.insert("बजे")

        # hour minute second
        graph_hms = (
            hour
            + delete_space
            + insert_bajkar
            + insert_space
            + minute
            + delete_space
            + insert_minute
            + insert_space
            + second
            + delete_space
            + insert_second
        )

        # hour minute
        graph_hm = hour + delete_space + insert_bajkar + insert_space + minute + delete_space + insert_minute

        # hour
        graph_h = hour + delete_space + insert_baje

        self.graph = graph_hms | graph_hm | graph_h

        final_graph = self.graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
