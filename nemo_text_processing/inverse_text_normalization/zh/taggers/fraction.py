# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. 二分之一 -> tokens { fraction { denominator: "2" numerator: "1"} }
        e.g. 五又二分之一 -> tokens { fraction { integer_part: "1" denominator: "2" numerator: "1" } }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        graph_cardinal = cardinal.just_cardinals
        integer_component = pynutil.insert('integer_part: "') + graph_cardinal + pynutil.insert('"')
        denominator_component = (
            pynutil.insert('denominator: "') + graph_cardinal + pynutil.delete("分之") + pynutil.insert('"')
        )
        numerator_component = pynutil.insert('numerator: "') + graph_cardinal + pynutil.insert('"')

        graph_only_fraction = denominator_component + pynutil.insert(" ") + numerator_component
        graph_fraction_with_int = integer_component + pynutil.delete("又") + pynutil.insert(" ") + graph_only_fraction

        graph_fraction = graph_only_fraction | graph_fraction_with_int

        final_graph = self.add_tokens(graph_fraction)
        self.fst = final_graph.optimize()
