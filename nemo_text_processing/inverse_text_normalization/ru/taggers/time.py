# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.ru.verbalizers.time import TimeFst as TNTimeVerbalizer
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "два часа пятнадцать минут" -> time { hours: "02:15" }

    Args:
        tn_time: Text Normalization Time graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_time: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        tn_time_tagger = tn_time.graph_preserve_order
        tn_time_verbalizer = TNTimeVerbalizer().graph
        tn_time_graph_preserve_order = pynini.compose(tn_time_tagger, tn_time_verbalizer).optimize()
        graph_preserve_order = pynini.invert(tn_time_graph_preserve_order).optimize()
        graph_preserve_order = pynutil.insert("hours: \"") + graph_preserve_order + pynutil.insert("\"")

        # "пятнадцать минут шестого" -> 17:15
        # Requires permutations for the correct verbalization
        m_next_h = (
            pynutil.insert("minutes: \"")
            + pynini.invert(tn_time.minutes).optimize()
            + pynutil.insert("\"")
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert("hours: \"")
            + pynini.invert(tn_time.increment_hour_ordinal).optimize()
            + pynutil.insert("\"")
        ).optimize()

        # "без пятнадцати минут шесть" -> 17:45
        # Requires permutation for the correct verbalization
        m_to_h = (
            pynini.cross("без ", "minutes: \"")
            + pynini.invert(tn_time.mins_to_h)
            + pynutil.insert("\"")
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert("hours: \"")
            + pynini.invert(tn_time.increment_hour_cardinal).optimize()
            + pynutil.insert("\"")
        )

        graph_reserve_order = m_next_h | m_to_h
        graph = graph_preserve_order | graph_reserve_order
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
