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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions, e.g.
    1/2 -> tokens { fraction { denominator: "二" numerator: "一"} }
    1と3/4 -> fraction { integer: "一" denominator: "四" numerator: "三" }
    一荷四分の三 -> fraction { integer: "1" denominator: "4" numerator: "3" }
    ルート三分の一 -> fraction { denominator: "√3" numerator: "1" }
    一点六五分の五十 -> fraction { denominator: "1.65" numerator: "50" }
    マイナス1/2 -> tokens { fraction { denominator: "二" numerator: "一"} }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal = cardinal.just_cardinals
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        slash = pynutil.delete('/')
        morphemes = pynini.accep('分の')
        root = pynini.accep('√')

        decimal_number = (
            cardinal
            + pynini.cross(".", "点")
            + pynini.closure(pynini.closure(graph_digit) | pynini.closure(graph_zero))
        )

        integer_component = (
            pynutil.insert('integer_part: \"')
            + (cardinal | (root + cardinal) | decimal_number | (root + decimal_number))
            + pynutil.insert("\"")
        )
        integer_component_with_char = (
            pynutil.insert('integer_part: \"')
            + (
                (cardinal | (root + cardinal) | decimal_number | (root + decimal_number))
                + (pynini.accep("と") | pynini.accep("荷"))
            )
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        denominator_component = (
            pynutil.insert("denominator: \"")
            + (cardinal | (root + cardinal) | decimal_number | (root + decimal_number))
            + pynutil.insert("\"")
        )
        numerator_component = (
            pynutil.insert("numerator: \"")
            + (cardinal | (root + cardinal) | decimal_number | (root + decimal_number))
            + pynutil.insert("\"")
        )

        # 3/4, 1 3/4, 1と3/4, -3/4, -1 3/4, 1と3/4, √1と3/4 and any combination of root number, cardinal number and decimal number
        graph_fraction_slash = (
            pynini.closure(
                (integer_component + pynini.accep(NEMO_SPACE))
                | (integer_component_with_char + pynutil.insert(NEMO_SPACE)),
                0,
                1,
            )
            + numerator_component
            + slash
            + pynutil.insert(NEMO_SPACE)
            + denominator_component
        )

        # 4分の3 -4分の3 and any combs
        graph_fraction_word = (
            pynini.closure(
                (
                    integer_component + pynini.accep(NEMO_SPACE)
                    | integer_component_with_char + pynutil.insert(NEMO_SPACE)
                ),
                0,
                1,
            )
            + denominator_component
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("morphosyntactic_features: \"")
            + morphemes
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + numerator_component
        )

        optional_sign = (
            pynutil.insert("negative: \"") + (pynini.accep("マイナス") | pynini.cross("-", "マイナス")) + pynutil.insert("\"")
        )

        graph_fraction_slash_sigh = pynini.closure(optional_sign + pynutil.insert(NEMO_SPACE), 0, 1) + (
            graph_fraction_slash | graph_fraction_word
        )

        graph = graph_fraction_slash_sigh  # |

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
