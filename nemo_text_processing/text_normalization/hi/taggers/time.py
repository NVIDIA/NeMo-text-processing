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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    HI_DEDH,
    HI_DHAI,
    HI_PAUNE,
    HI_SADHE,
    HI_SAVVA,
    MIN_NEG_WEIGHT,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

# Time patterns specific to time tagger - support both Devanagari and Arabic digits
HI_DOUBLE_ZERO = pynini.union("००", "00")
HI_TIME_FIFTEEN = pynini.union(":१५", ":15")
HI_TIME_THIRTY = pynini.union(":३०", ":30")
HI_TIME_FORTYFIVE = pynini.union(":४५", ":45")

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

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        delete_colon = pynutil.delete(":")
        delete_leading_zero = pynini.closure(pynutil.delete("0") | pynutil.delete("०"), 0, 1)
        cardinal_graph = delete_leading_zero + (cardinal.digit | cardinal.teens_and_ties)

        self.hours = pynutil.insert("hours: \"") + delete_leading_zero + hours_graph + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minutes_graph + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + seconds_graph + pynutil.insert("\" ")

       # hour minute seconds (allows 00 minutes and 00 seconds)
        graph_hms = (
            self.hours + delete_colon + insert_space + self.minutes + delete_colon + insert_space + self.seconds
        )

        # Restrict graph_hm from accepting 00 minutes so H:00 falls back to graph_h
        exclude_double_zero = pynini.union("00", "००").optimize()
        minutes_no_zero_graph = pynini.difference(pynini.project(minutes_graph, "input"), exclude_double_zero) @ minutes_graph
        hm_minutes_restricted = pynutil.insert("minutes: \"") + minutes_no_zero_graph + pynutil.insert("\" ")

        # hour minute
        graph_hm = self.hours + delete_colon + insert_space + hm_minutes_restricted

        # hour
        graph_h = self.hours + delete_colon + pynutil.delete(HI_DOUBLE_ZERO)

        # Support all combinations of Devanagari and Arabic digits for dedh/dhai patterns
        dedh_dhai_graph = delete_leading_zero + pynini.string_map(
            [
                ("१:३०", HI_DEDH),
                ("१:30", HI_DEDH),
                ("1:३०", HI_DEDH),
                ("1:30", HI_DEDH),
                ("२:३०", HI_DHAI),
                ("२:30", HI_DHAI),
                ("2:३०", HI_DHAI),
                ("2:30", HI_DHAI),
            ]
        )

        savva_numbers = cardinal_graph + pynini.cross(HI_TIME_FIFTEEN, "")
        savva_graph = pynutil.insert(HI_SAVVA) + pynutil.insert(NEMO_SPACE) + savva_numbers

        # Restrict 'sadhe' from accepting 1 or 2 so it doesn't conflict with dedh/dhai
        exclude_tsv = pynini.string_file(get_abs_path("data/time/exclude_dedh_dhai.tsv"))
        exclude_dedh_dhai = pynini.project(exclude_tsv, "input").optimize()
        
        # Project cardinal_graph to an acceptor, subtract exceptions, then compose (@) back to the transducer
        valid_sadhe_inputs = pynini.difference(pynini.project(cardinal_graph, "input"), exclude_dedh_dhai)
        sadhe_cardinal = valid_sadhe_inputs @ cardinal_graph

        sadhe_numbers = sadhe_cardinal + pynini.cross(HI_TIME_THIRTY, "")
        sadhe_graph = pynutil.insert(HI_SADHE) + pynutil.insert(NEMO_SPACE) + sadhe_numbers

        paune = delete_leading_zero + pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(HI_TIME_FORTYFIVE, "")
        paune_graph = pynutil.insert(HI_PAUNE) + pynutil.insert(NEMO_SPACE) + paune_numbers

        graph_dedh_dhai = (
            pynutil.insert("morphosyntactic_features: \"")
            + dedh_dhai_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_savva = (
            pynutil.insert("morphosyntactic_features: \"")
            + savva_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_sadhe = (
            pynutil.insert("morphosyntactic_features: \"")
            + sadhe_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        graph_paune = (
            pynutil.insert("morphosyntactic_features: \"")
            + paune_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        arabic_1_2 = pynini.closure(NEMO_DIGIT, 1, 2)
        arabic_2 = pynini.closure(NEMO_DIGIT, 2, 2)
        arabic_valid_time = (
            arabic_1_2 + pynini.accep(":") + arabic_2 + pynini.closure(pynini.accep(":") + arabic_2, 0, 1)
        )

        deva_1_2 = pynini.closure(NEMO_HI_DIGIT, 1, 2)
        deva_2 = pynini.closure(NEMO_HI_DIGIT, 2, 2)
        deva_valid_time = deva_1_2 + pynini.accep(":") + deva_2 + pynini.closure(pynini.accep(":") + deva_2, 0, 1)

        valid_time_pattern = pynini.union(arabic_valid_time, deva_valid_time).optimize()

        # 2. Give special patterns a minimal negative weight so they safely beat the fallback graph_hm
        unfiltered_graph = (
            graph_hms
            | pynutil.add_weight(graph_hm, 0.3)
            | pynutil.add_weight(graph_h, 0.3)
            | pynutil.add_weight(graph_dedh_dhai, MIN_NEG_WEIGHT)
            | pynutil.add_weight(graph_savva, MIN_NEG_WEIGHT)
            | pynutil.add_weight(graph_sadhe, MIN_NEG_WEIGHT)
            | pynutil.add_weight(graph_paune, MIN_NEG_WEIGHT)
        )

        final_graph = pynini.compose(valid_time_pattern, unfiltered_graph)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
