# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import GraphFst, delete_extra_space, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Telugu fractions.

    Examples:
        రెండు బై మూడు
        -> fraction { numerator: "౨" denominator: "౩" }

        రెండు మరియు ఒకటి బై మూడు
        -> fraction { integer_part: "౨" numerator: "౧" denominator: "౩" }

        అర
        -> fraction { numerator: "౧" denominator: "౨" }

        ఒకటిన్నర
        -> fraction { integer_part: "౧" numerator: "౧" denominator: "౨" }

    Args:
        cardinal: CardinalFst instance used to obtain the raw cardinal graph.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        graph_cardinal = cardinal.graph_no_exception

        integer = pynutil.insert('integer_part: "') + graph_cardinal + pynutil.insert('"')

        numerator = pynutil.insert('numerator: "') + graph_cardinal + pynutil.insert('"')

        denominator = pynutil.insert(' denominator: "') + graph_cardinal + pynutil.insert('"')

        delete_bai = delete_extra_space + pynutil.delete("బై") + delete_space

        delete_mariyu = delete_extra_space + pynutil.delete("మరియు") + delete_space

        graph_fraction = numerator + delete_bai + denominator

        graph_half = pynutil.delete("అర") + pynutil.insert('numerator: "౧" denominator: "౨"')

        graph_quarter = pynutil.delete("పావు") + pynutil.insert('numerator: "౧" denominator: "౪"')

        graph_three_quarter = pynutil.delete("ముప్పావు") + pynutil.insert('numerator: "౩" denominator: "౪"')

        graph_lexical_fraction = (graph_half | graph_quarter | graph_three_quarter).optimize()

        graph_mixed_fraction = integer + delete_mariyu + (graph_fraction | graph_lexical_fraction)

        graph_one_half = pynutil.delete("ఒకటిన్నర") + pynutil.insert('integer_part: "౧" numerator: "౧" denominator: "౨"')

        graph_two_half = pynutil.delete("రెండున్నర") + pynutil.insert(
            'integer_part: "౨" numerator: "౧" denominator: "౨"'
        )

        graph_three_half = pynutil.delete("మూడున్నర") + pynutil.insert(
            'integer_part: "౩" numerator: "౧" denominator: "౨"'
        )

        graph_eight_half = pynutil.delete("ఎనిమిదిన్నర") + pynutil.insert(
            'integer_part: "౮" numerator: "౧" denominator: "౨"'
        )

        graph_spoken_mixed = (graph_one_half | graph_two_half | graph_three_half | graph_eight_half).optimize()

        graph = (graph_fraction | graph_mixed_fraction | graph_lexical_fraction | graph_spoken_mixed).optimize()

        self.fst = self.add_tokens(graph).optimize()
