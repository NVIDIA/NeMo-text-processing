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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal, e.g.
        5-րդ -> ordinal { integer: "հինգերորդ" }
        1-ին -> ordinal { integer: "առաջին" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.all_nums_no_tokens
        cardinal_format = pynini.closure(NEMO_DIGIT | pynini.accep(","))
        rd = pynini.accep("-րդ")
        first_format = (
            pynini.closure(cardinal_format + (NEMO_DIGIT - "1"), 0, 1) + pynini.accep("1") + pynutil.delete("-ին")
        )
        second_format = pynini.closure(cardinal_format + (NEMO_DIGIT - "2"), 0, 1) + pynini.accep("2")
        third_format = pynini.closure(cardinal_format + (NEMO_DIGIT - "1"), 0, 1) + pynini.accep("3")
        fourth_format = pynini.closure(cardinal_format + (NEMO_DIGIT - "1"), 0, 1) + pynini.accep("4")
        th_format = pynini.closure(
            (NEMO_DIGIT - "1" - "2" - "3" - "4") | (cardinal_format + "1" + NEMO_DIGIT) | cardinal_format, 1
        )

        first = pynini.cross("1", "առաջին")
        second = pynini.cross("2", "երկրորդ")
        third = pynini.cross("3", "երրորդ")
        fourth = pynini.cross("4", "չորրորդ")

        special_denominator_graph = second_format @ second | third_format @ third | fourth_format @ fourth

        self.denominator_graph = (
            pynutil.add_weight(first_format @ first, 1)
            | pynutil.add_weight(special_denominator_graph, 1)
            | pynutil.add_weight(th_format @ cardinal_graph + pynutil.insert("երորդ"), 1.5)
        ).optimize()

        special_ordinals_graph = (
            (second_format + pynutil.delete(rd)) @ second
            | (third_format + pynutil.delete(rd)) @ third
            | (fourth_format + pynutil.delete(rd)) @ fourth
        )

        self.graph = (
            pynutil.add_weight(first_format @ first, 1)
            | pynutil.add_weight(special_ordinals_graph, 1)
            | pynutil.add_weight((th_format + pynutil.delete(rd)) @ cardinal_graph + pynutil.insert("երորդ"), 1.5)
        ).optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
