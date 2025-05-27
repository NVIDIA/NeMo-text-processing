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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        १२:३०:३०  -> time { hours: "बारह" minutes: "तीस" seconds: "तीस" }
        १:४०  -> time { hours: "एक" minutes: "चालीस" }
        १:००  -> time { hours: "एक" }

    Args:
        time: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal:GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        cardinal_graph = cardinal.digit | cardinal.teens_and_ties

        self.hours = pynutil.insert("hours: \"") + hours_graph + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minutes_graph + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + seconds_graph + pynutil.insert("\" ")

        # hour minute seconds
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + self.minutes

        # hour
        graph_h = self.hours + delete_colon + pynutil.delete("००")

        dedh_dhai_graph = pynini.string_map([("१:३०", "डेढ़"), ("२:३०", "ढाई")])

        savva_numbers = cardinal_graph + pynini.cross(":१५", "")
        savva_graph = pynutil.insert("सवा ") + savva_numbers

        sadhe_numbers = cardinal_graph + pynini.cross(":३०", "")
        sadhe_graph = pynutil.insert("साढ़े ") + sadhe_numbers

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(":४५", "")
        paune_graph = pynutil.insert("पौने ") + paune_numbers

        graph_dedh_dhai = (
            pynutil.insert("morphosyntactic_features: \"")
            + dedh_dhai_graph
            + pynutil.insert("\" ")
        )

        graph_savva = (
            pynutil.insert("morphosyntactic_features: \"")
            + savva_graph
            + pynutil.insert("\" ")
        )

        graph_sadhe = (
            pynutil.insert("morphosyntactic_features: \"")
            + sadhe_graph
            + pynutil.insert("\" ")
        )

        graph_paune = (
            pynutil.insert("morphosyntactic_features: \"")
            + paune_graph
            + pynutil.insert("\" ")
        )

        final_graph = (
            graph_hms 
            | pynutil.add_weight(graph_hm, 0.01)
            | pynutil.add_weight(graph_h, 0.01)
            | pynutil.add_weight(graph_dedh_dhai, 0.001)
            | pynutil.add_weight(graph_savva, 0.005)
            | pynutil.add_weight(graph_sadhe, 0.005)
            | pynutil.add_weight(graph_paune, 0.001)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
