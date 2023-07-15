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
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction, e.g.,
        1/2 -> tokens { fraction { denominator: "二" numerator: "一"} }
        5又1/2 -> tokens { fraction { integer_part: "五" denominator: "二" numerator: "一" } }
        5又2分之1 -> tokens { {} }
        2分之1 -> tokens { fraction { denominator: "二" numerator: "一"} }
        100分之1 -> tokens { fraction { denominator: "一百" numerator: "一"} }
        百分之1 -> tokens { fraction { denominator: "百" numerator: "一"} }
        98% -> tokens { fraction { denominator: "百" numerator: "九十八"} }
    
    Args:
        cardinal: CardinalFst， decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        graph_cardinals = cardinal.just_cardinals
        graph_decimal = decimal.decimal

        slash = pynutil.delete('/')
        morpheme = pynutil.delete('分之')
        suffix = pynini.union(
            "百",
            "千",
            "万",
            "十万",
            "百万",
            "千万",
            "亿",
            "十亿",
            "百亿",
            "千亿",
            "萬",
            "十萬",
            "百萬",
            "千萬",
            "億",
            "十億",
            "百億",
            "千億",
            "拾萬",
            "佰萬",
            "仟萬",
            "拾億",
            "佰億",
            "仟億",
            "拾万",
            "佰万",
            "仟万",
            "仟亿",
            "佰亿",
            "仟亿",
        )

        integer_component = pynutil.insert("integer_part: \"") + graph_cardinals + pynutil.insert("\"")
        denominator_component = pynutil.insert("denominator: \"") + graph_cardinals + pynutil.insert("\"")
        numerator_component = pynutil.insert("numerator: \"") + graph_cardinals + pynutil.insert("\"")

        graph_with_integer = (
            pynini.closure(integer_component + pynutil.delete('又'), 0, 1)
            + pynutil.insert(' ')
            + numerator_component
            + slash
            + pynutil.insert(' ')
            + denominator_component
        )
        graph_only_slash = numerator_component + slash + pynutil.insert(' ') + denominator_component

        graph_morpheme = (denominator_component + morpheme + pynutil.insert(' ') + numerator_component) | (
            integer_component
            + pynutil.delete('又')
            + pynutil.insert(' ')
            + denominator_component
            + morpheme
            + pynutil.insert(' ')
            + numerator_component
        )

        graph_with_suffix = (
            pynini.closure(pynutil.insert("denominator: \"") + suffix + pynutil.insert("\""), 0, 1)
            + morpheme
            + pynutil.insert(' ')
            + numerator_component
        )

        percentage = pynutil.delete('%')
        graph_percentage = (
            numerator_component
            + percentage
            + pynutil.insert(' ')
            + pynutil.insert("denominator: \"百")
            + pynutil.insert("\"")
        )

        graph_hundred = pynutil.delete('100%') + pynutil.insert('numerator: \"百\" denominator: \"百"')

        graph_optional_sign = (
            (pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"负\"")))
            | (pynini.closure(pynutil.insert("positive: ") + pynini.cross("+", "\"正\"")))
            | (pynutil.insert("positive: ") + pynutil.insert("\"") + pynini.accep('正') + pynutil.insert("\""))
            | (
                pynutil.insert('negative: ')
                + pynutil.insert("\"")
                + (pynini.accep('负') | pynini.cross('負', '负'))
                + pynutil.insert("\"")
            )
        )

        graph_decimals = (
            graph_decimal
            + pynutil.insert(" ")
            + percentage
            + pynutil.insert("denominator: \"百")
            + pynutil.insert("\"")
        )

        graph = (
            graph_with_integer
            | graph_only_slash
            | graph_morpheme
            | graph_with_suffix
            | graph_percentage
            | graph_decimals
            | pynutil.add_weight(graph_hundred, -3.0)
        )
        graph_with_sign = (
            (graph_optional_sign + pynutil.insert(" ") + graph_with_integer)
            | (graph_optional_sign + pynutil.insert(" ") + graph_only_slash)
            | (graph_optional_sign + pynutil.insert(" ") + graph_morpheme)
            | (graph_optional_sign + pynutil.insert(" ") + graph_with_suffix)
            | (graph_optional_sign + pynutil.insert(" ") + graph_percentage)
            | pynutil.add_weight((graph_optional_sign + pynutil.insert(" ") + graph_hundred), -3.0)
        )

        final_graph = graph | graph_with_sign

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
