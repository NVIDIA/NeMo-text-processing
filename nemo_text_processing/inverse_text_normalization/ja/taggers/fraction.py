# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import GraphFst


class FractionFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        """
        Fitite state transducer for classifying fractions
        e.g.,
        四分の三 -> fraction { denominator: "4" numerator: "3" }
        一と四分の三 -> fraction { integer: "1" denominator: "4" numerator: "3" }
        一荷四分の三 -> fraction { integer: "1" denominator: "4" numerator: "3" }
        ルート三分の一 -> fraction { denominator: "√3" numerator: "1" }
        一点六五分の五十 -> fraction { denominator: "1.65" numerator: "50" }
        二ルート六分の三 -> -> fraction { denominator: "2√6 " numerator: "3" }
        """
        super().__init__(name="fraction", kind="classify")

        cardinal = cardinal.just_cardinals
        decimal = decimal.just_decimal

        fraction_word = pynutil.delete("分の") | pynutil.delete(" 分 の　") | pynutil.delete("分 の　") | pynutil.delete("分 の")

        integer_word = pynutil.delete("と") | pynutil.delete("荷")
        root_word = pynini.accep("√") | pynini.cross("ルート", "√")

        graph_sign = (
            pynutil.insert("negative: \"") + (pynini.accep("-") | pynini.cross("マイナス", "-")) + pynutil.insert("\"")
        )

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + (
                (decimal | decimal + integer_word)
                | ((cardinal + root_word + cardinal) | (cardinal + root_word + cardinal + integer_word))
                | ((root_word + cardinal) | (root_word + cardinal + integer_word))
                | (cardinal | (cardinal + integer_word))
            )
            + pynutil.insert("\"")
        )

        graph_denominator = (
            pynutil.insert("denominator: \"")
            + (
                ((decimal) | (cardinal + root_word + cardinal) | (root_word + cardinal) | cardinal)
                + pynini.closure(pynutil.delete(' '), 0, 1)
            )
            + pynutil.insert("\"")
        )

        graph_numerator = (
            pynutil.insert("numerator: \"")
            + (
                pynini.closure(pynutil.delete(' '))
                + ((decimal) | (cardinal + root_word + cardinal) | (root_word + cardinal) | cardinal)
            )
            + pynutil.insert("\"")
        )

        graph_fraction_sign = (
            graph_sign
            + pynutil.insert(" ")
            + graph_denominator
            + pynutil.insert(" ")
            + fraction_word
            + graph_numerator
        )

        graph_fraction_no_sign = graph_denominator + pynutil.insert(" ") + fraction_word + graph_numerator

        graph_regular_fractions = graph_fraction_sign | graph_fraction_no_sign

        graph_integer_fraction_sign = (
            pynini.closure((graph_sign + pynutil.insert(" ")), 0, 1)
            + pynutil.add_weight(graph_integer, 1.1)
            + pynutil.insert(" ")
            + graph_denominator
            + pynutil.insert(" ")
            + fraction_word
            + graph_numerator
        )

        final_graph = graph_regular_fractions | graph_integer_fraction_sign

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
