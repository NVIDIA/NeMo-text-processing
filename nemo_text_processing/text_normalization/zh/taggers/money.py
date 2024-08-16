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


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying moneys, e.g.
    '$23' -> money { integer: "二十三" currency: "美元" }
    '23美元' -> money { integer: "二十三" currency: "美元" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal = cardinal.just_cardinals

        currency = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        currency_mandarin = pynini.string_file(get_abs_path("data/money/currency_mandarin.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        # regular money gramamr with currency symbols $1000
        currency_component = pynutil.insert("currency: \"") + currency + pynutil.insert("\"")
        number_component = pynutil.insert("integer_part: \"") + (cardinal | (cardinal + suffix)) + pynutil.insert("\"")
        graph_regular_money = currency_component + pynutil.insert(" ") + number_component

        # 块 元 毛 with optional symbols
        unit_major = (
            pynini.accep("块")
            | pynini.accep("元")
            | pynini.closure(pynini.cross("塊", "块"), 1)
            | pynini.closure(pynini.cross("塊錢", "块钱"), 1)
            | pynini.accep("块钱")
        )
        unit_minor = pynini.accep("角") | pynini.accep("毛")
        unit_minor_alt = pynini.accep("分")
        currency_mandarin_component = pynutil.insert("currency: \"") + currency_mandarin + pynutil.insert("\"")
        unit_components = (
            (pynutil.insert("currency: \"") + unit_major + pynutil.insert("\""))
            | (pynutil.insert("currency_maj: \"") + unit_minor + pynutil.insert("\""))
            | (pynutil.insert("currency_min: \"") + unit_minor_alt + pynutil.insert("\""))
        )

        graph_unit_only = (
            number_component
            + pynutil.insert(" ")
            + unit_components
            + pynini.closure(pynutil.insert(" ") + currency_mandarin_component, 0, 1)
        )
        graph_units = pynini.closure(graph_unit_only, 1, 3)

        # only currency part as mandarins
        graph_mandarin_money = number_component + pynutil.insert(" ") + currency_mandarin_component

        # larger money as decimals
        graph_decimal = (
            pynutil.insert('integer_part: \"')
            + (
                pynini.closure(cardinal, 1)
                + pynutil.delete('.')
                + pynutil.insert('点')
                + pynini.closure((graph_digit | graph_zero), 1)
            )
            + pynutil.insert("\"")
        )
        graph_decimal_money = (
            pynini.closure(graph_decimal, 1)
            + pynini.closure((pynutil.insert(' quantity: \"') + suffix + pynutil.insert('\"')), 0, 1)
            + pynutil.insert(" ")
            + pynini.closure(currency_mandarin_component, 1)
        ) | (
            pynini.closure(currency_component, 1)
            + pynutil.insert(" ")
            + pynini.closure(graph_decimal, 1)
            + pynini.closure(
                (pynutil.insert(" ") + pynutil.insert('quantity: \"') + suffix + pynutil.insert('\"')), 0, 1
            )
        )

        graph = (
            graph_regular_money
            | graph_units
            | pynutil.add_weight(graph_mandarin_money, -3.0)
            | pynutil.add_weight(graph_decimal_money, -1.0)
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
