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

from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal, e.g.
    第100 -> ordinal { integer: "第一百" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)

        graph_cardinal = cardinal.just_cardinals
        morpheme = pynini.accep('第')
        graph_ordinal = morpheme + graph_cardinal
        graph_ordinal_final = pynutil.insert("integer: \"") + graph_ordinal + pynutil.insert("\"")

        # range
        range_source = pynini.accep("从")
        range_goal = (
            pynini.accep("-")
            | pynini.accep("~")
            | pynini.accep("——")
            | pynini.accep("—")
            | pynini.accep("到")
            | pynini.accep("至")
        )
        graph_range_source = (
            pynini.closure((pynutil.insert("range: \"") + range_source + pynutil.insert("\" ")), 0, 1)
            + pynutil.insert("integer: \"")
            + graph_ordinal
            + pynutil.insert("\"")
            + pynutil.insert(" range: \"")
            + range_goal
            + pynutil.insert("\" ")
            + pynutil.insert("integer: \"")
            + (graph_ordinal | graph_cardinal)
            + pynutil.insert("\"")
        )
        graph_range_goal = (
            pynutil.insert("integer: \"")
            + graph_ordinal
            + pynutil.insert("\"")
            + pynutil.insert(" range: \"")
            + range_goal
            + pynutil.insert("\" ")
            + pynutil.insert("integer: \"")
            + (graph_ordinal | graph_cardinal)
            + pynutil.insert("\"")
        )
        graph_range_final = graph_range_source | graph_range_goal

        final_graph = graph_ordinal_final | graph_range_final

        graph_ordinal_final = self.add_tokens(final_graph)
        self.fst = graph_ordinal_final.optimize()
