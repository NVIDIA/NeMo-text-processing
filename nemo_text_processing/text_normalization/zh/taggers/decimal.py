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
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


def get_quantity(decimal):
    suffix = pynini.union(
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
        "万亿",
        "萬億",
    )
    res = decimal + pynutil.insert(" quantity: \"") + suffix + pynutil.insert("\"")

    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        0.5 -> decimal { integer_part: "零" fractional_part: "五" }
        0.5万 -> decimal { integer_part: "零" fractional_part: "五" quantity: "万" }
        -0.5万 -> decimal { negative: "负" integer_part: "零" fractional_part: "五" quantity: "万"}

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_before_decimal = cardinal.just_cardinals
        cardinal_after_decimal = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        graph_integer = pynutil.insert('integer_part: \"') + cardinal_before_decimal + pynutil.insert("\"")

        graph_fraction = (
            pynutil.insert("fractional_part: \"")
            + pynini.closure((pynini.closure(cardinal_after_decimal, 1) | (pynini.closure(zero, 1))), 1)
            + pynutil.insert("\"")
        )
        graph_decimal = graph_integer + pynutil.delete('.') + pynutil.insert(" ") + graph_fraction
        self.regular_decimal = graph_decimal.optimize()

        graph_sign = (
            (
                pynini.closure(pynutil.insert("negative: \"") + pynini.cross("-", "负"))
                + pynutil.insert("\"")
                + pynutil.insert(" ")
            )
        ) | (
            (
                pynutil.insert('negative: ')
                + pynutil.insert("\"")
                + (pynini.accep('负') | pynini.cross('負', '负'))
                + pynutil.insert("\"")
                + pynutil.insert(' ')
            )
        )
        graph_with_sign = graph_sign + graph_decimal
        graph_regular = graph_with_sign | graph_decimal

        graph_decimal_quantity = get_quantity(graph_decimal)
        graph_sign_quantity = graph_sign + graph_decimal_quantity
        graph_quantity = graph_decimal_quantity | graph_sign_quantity

        final_graph = graph_regular | graph_quantity
        self.decimal = final_graph.optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
