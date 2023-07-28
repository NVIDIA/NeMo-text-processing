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
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
    '23' -> cardinal { integer: "二十三" }
    -10000 -> cardinal { negative: "负" integer: "一万" }
    +10000 -> cardinal { positive: "正" integer: "一万" }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # imports
        zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        digit_tens = pynini.string_file(get_abs_path("data/number/digit_tens.tsv"))

        # morphemes inserted + punctuation
        tens_digit = pynutil.insert('十')
        hundred_digit = pynutil.insert('百')
        thousand_digit = pynutil.insert('千')
        tenthousand_digit = pynutil.insert('万')
        hundredmillion_digit = pynutil.insert('亿')
        delete_punct = pynini.closure(pynutil.delete(',') | pynutil.delete('，'))

        # 十几; 10-19
        graph_teen = (
            pynini.closure(delete_punct)
            + pynini.cross('1', '十')
            + (
                (pynini.closure(delete_punct) + (pynini.closure(delete_punct) + digit))
                | (pynini.closure(delete_punct) + pynini.cross('0', ''))
            )
        )

        # 十几; 10-19 but when not alone, but within a larger number, (e.g, 119)
        graph_teen_alt = (
            (pynini.closure(delete_punct) + (pynini.cross('1', '一十') + pynini.closure(delete_punct) + digit))
            | (pynini.closure(delete_punct) + pynini.cross('10', '一十'))
            | (pynini.closure(delete_punct) + (pynini.cross('1,0', '一十') | pynini.cross('1，0', '一十')))
        )  # when the teen is not by itself but with in a larger number

        # 几十; 20-99
        graph_tens = (
            pynini.closure(delete_punct)
            + (digit_tens + tens_digit + pynini.closure(delete_punct) + ((pynini.closure(delete_punct) + digit)))
        ) | (
            digit_tens + tens_digit + (pynini.closure(delete_punct) + (pynini.cross('0', '') | pynini.cross(',0', '')))
        )

        # 百; 100-999; hundreds
        graph_hundred = (
            (
                digit
                + (
                    pynutil.delete('00')
                    | (pynutil.delete(',00') | pynutil.delete('，00'))
                    | (pynutil.delete('0,0') | pynutil.delete('0，0'))
                )
                + hundred_digit
            )
            | (digit + hundred_digit + (graph_tens | graph_teen_alt))
            | (
                digit
                + hundred_digit
                + (
                    (pynini.cross(',0', '零') | pynini.cross('，0', '零'))
                    | pynini.cross('0', '零')
                    | (pynini.cross('0,', '零') | pynini.cross('0，', '零'))
                )
                + digit
            )
        )

        # 千; 1000-9999; thousands
        graph_thousand = (
            (
                digit
                + (
                    (pynutil.delete(',000') | pynutil.delete('000') | pynutil.delete('0,00') | pynutil.delete('00,0'))
                    | (
                        pynutil.delete('，000')
                        | pynutil.delete('000')
                        | pynutil.delete('0，00')
                        | pynutil.delete('00，0')
                    )
                )
                + thousand_digit
            )
            | (digit + pynini.closure(delete_punct) + thousand_digit + graph_hundred)
            | (
                digit
                + thousand_digit
                + (pynini.cross('0', '零') | ((pynini.cross(',0', '零') | pynini.cross('，0', '零'))))
                + (graph_tens | graph_teen_alt)
            )
            | (
                digit
                + pynini.closure(delete_punct)
                + thousand_digit
                + (
                    pynini.cross('00', '零')
                    | (pynini.cross(',00', '零') | pynini.cross('，00', '零'))
                    | (pynini.cross('0,0', '零') | pynini.cross('0，0', '零'))
                    | (pynini.cross('00,', '零') | pynini.cross('00，', '零'))
                )
                + digit
            )
        )

        # 万; 10000-99999; ten thousands
        graph_tenthousand = (
            (
                digit
                + (pynutil.delete('0000') | (pynutil.delete('0,000') | pynutil.delete('0，000')))
                + tenthousand_digit
            )
            | (digit + tenthousand_digit + graph_thousand)
            | (
                digit
                + tenthousand_digit
                + (pynini.cross('0', '零') | (pynini.cross('0,', '零') | pynini.cross('0，', '零')))
                + graph_hundred
            )
            | (
                digit
                + tenthousand_digit
                + (pynini.cross('00', '零') | (pynini.cross('0,0', '零') | pynini.cross('0，0', '零')))
                + (graph_tens | graph_teen_alt)
            )
            | (
                digit
                + tenthousand_digit
                + (pynini.cross('000', '零') | (pynini.cross('0,00', '零') | pynini.cross('0，00', '零')))
                + digit
            )
        )

        # 十万; 100000-999999; hundred thousands
        graph_hundredthousand = (
            pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + tenthousand_digit
                    + (pynutil.delete('0000') | (pynutil.delete('0,000') | pynutil.delete('0，000')))
                ),
                -0.1,
            )
            | ((graph_tens | graph_teen) + tenthousand_digit + graph_thousand)
            | (
                (graph_tens | graph_teen)
                + tenthousand_digit
                + (pynini.cross('0', '零') | (pynini.cross('0,', '零') | pynini.cross('0，', '零')))
                + graph_hundred
            )
            | (
                (graph_tens | graph_teen)
                + tenthousand_digit
                + (pynini.cross('00', '零') | (pynini.cross('0,0', '零') | pynini.cross('0，0', '零')))
                + (graph_tens | graph_teen_alt)
            )
            | (
                (graph_tens | graph_teen)
                + tenthousand_digit
                + (pynini.cross('000', '零') | (pynini.cross('0,00', '零') | pynini.cross('0，00', '零')))
                + digit
            )
        )

        # 百万; 1000000-9999999; millions
        graph_million = (
            pynutil.add_weight(
                (
                    graph_hundred
                    + tenthousand_digit
                    + (pynutil.delete('0000') | (pynutil.delete('0,000') | pynutil.delete('0，000')))
                ),
                -1.0,
            )
            | (graph_hundred + tenthousand_digit + graph_thousand)
            | (
                graph_hundred
                + tenthousand_digit
                + (pynini.cross('0', '零') | (pynini.cross('0,', '零') | pynini.cross('0，', '零')))
                + graph_hundred
            )
            | (
                graph_hundred
                + tenthousand_digit
                + (pynini.cross('00', '零') | (pynini.cross('0,0', '零') | pynini.cross('0，0', '零')))
                + (graph_tens | graph_teen_alt)
            )
            | (
                graph_hundred
                + tenthousand_digit
                + (pynini.cross('000', '零') | (pynini.cross('0,00', '零') | pynini.cross('0，00', '零')))
                + digit
            )
        )

        # 千万; 10000000-99999999; ten millions
        graph_tenmillion = (
            pynutil.add_weight(
                (
                    graph_thousand
                    + (pynutil.delete('0000') | (pynutil.delete('0,000') | pynutil.delete('0，000')))
                    + tenthousand_digit
                ),
                -1.0,
            )
            | (graph_thousand + tenthousand_digit + graph_thousand)
            | (
                graph_thousand
                + tenthousand_digit
                + (pynini.cross('0', '零') | (pynini.cross('0,', '零') | pynini.cross('0，', '零')))
                + graph_hundred
            )
            | (
                graph_thousand
                + tenthousand_digit
                + (pynini.cross('00', '零') | (pynini.cross('0,0', '零') | pynini.cross('0，0', '零')))
                + (graph_tens | graph_teen_alt)
            )
            | (
                graph_thousand
                + tenthousand_digit
                + (pynini.cross('000', '零') | (pynini.cross('0,00', '零') | pynini.cross('0，00', '零')))
                + digit
            )
        )

        # 亿; 100000000-999999999; hundred millions
        graph_hundredmillion = (
            pynutil.add_weight(
                (
                    digit
                    + (pynutil.delete('00000000') | (pynutil.delete('00,000,000') | pynutil.delete('00，000，000')))
                    + hundredmillion_digit
                ),
                -2.0,
            )
            | pynutil.add_weight((digit + hundredmillion_digit + graph_tenmillion), -1.9)
            | pynutil.add_weight((digit + hundredmillion_digit + pynutil.delete('0') + graph_million), -1.8)
            | pynutil.add_weight(
                (digit + hundredmillion_digit + pynutil.delete('00') + pynutil.insert('零') + graph_hundredthousand),
                -1.7,
            )
            | pynutil.add_weight(
                (
                    digit
                    + hundredmillion_digit
                    + (pynutil.delete('000') | (pynutil.delete('00,0') | pynutil.delete('00，0')))
                    + pynutil.insert('零')
                    + graph_tenthousand
                ),
                -1.6,
            )
            | pynutil.add_weight(
                (
                    digit
                    + hundredmillion_digit
                    + (pynutil.delete('0000') | (pynutil.delete('00,00') | pynutil.delete('00，00')))
                    + pynutil.insert('零')
                    + graph_thousand
                ),
                -1.5,
            )
            | pynutil.add_weight(
                (
                    digit
                    + hundredmillion_digit
                    + (pynutil.delete('00000') | (pynutil.delete('00,000,') | pynutil.delete('00，000，')))
                    + pynutil.insert('零')
                    + graph_hundred
                ),
                -1.4,
            )
            | pynutil.add_weight(
                (
                    digit
                    + hundredmillion_digit
                    + (pynutil.delete('000000') | (pynutil.delete('00,000,0') | pynutil.delete('00，000，0')))
                    + pynutil.insert('零')
                    + (graph_tens | graph_teen_alt)
                ),
                -1.3,
            )
            | pynutil.add_weight(
                (
                    digit
                    + hundredmillion_digit
                    + (pynutil.delete('0000000') | (pynutil.delete('00,000,00') | pynutil.delete('00，000，00')))
                    + pynutil.insert('零')
                    + digit
                ),
                -1.2,
            )
        )

        # 十亿; 1000000000-9999999999; billions
        graph_billion = (
            pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + (pynutil.delete('00000000') | (pynutil.delete('00,000,000') | pynutil.delete('00，000，000')))
                    + hundredmillion_digit
                ),
                -2.0,
            )
            | pynutil.add_weight(((graph_tens | graph_teen) + hundredmillion_digit + graph_tenmillion), -1.9)
            | pynutil.add_weight(
                ((graph_tens | graph_teen) + hundredmillion_digit + pynutil.delete('0') + graph_million), -1.8
            )
            | pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + hundredmillion_digit
                    + pynutil.delete('00')
                    + pynutil.insert('零')
                    + graph_hundredthousand
                ),
                -1.7,
            )
            | pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + hundredmillion_digit
                    + (pynutil.delete('000') | (pynutil.delete('00,0') | pynutil.delete('00，0')))
                    + pynutil.insert('零')
                    + graph_tenthousand
                ),
                -1.6,
            )
            | pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + hundredmillion_digit
                    + (pynutil.delete('0000') | (pynutil.delete('00,00') | pynutil.delete('00，00')))
                    + pynutil.insert('零')
                    + graph_thousand
                ),
                -1.5,
            )
            | pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + hundredmillion_digit
                    + (pynutil.delete('00000') | (pynutil.delete('00,000,') | pynutil.delete('00，000，')))
                    + pynutil.insert('零')
                    + graph_hundred
                ),
                -1.4,
            )
            | pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + hundredmillion_digit
                    + (pynutil.delete('000000') | (pynutil.delete('00,000,0') | pynutil.delete('00，000，0')))
                    + pynutil.insert('零')
                    + (graph_tens | graph_teen_alt)
                ),
                -1.3,
            )
            | pynutil.add_weight(
                (
                    (graph_tens | graph_teen)
                    + hundredmillion_digit
                    + (pynutil.delete('0000000') | (pynutil.delete('00,000,00') | pynutil.delete('00，000，00')))
                    + pynutil.insert('零')
                    + digit
                ),
                -1.2,
            )
        )

        # 百亿; 10000000000-99999999999; ten billions
        graph_tenbillion = (
            pynutil.add_weight(
                (
                    graph_hundred
                    + (pynutil.delete('00000000') | (pynutil.delete('00,000,000') | pynutil.delete('00，000，000')))
                    + hundredmillion_digit
                ),
                -2.0,
            )
            | pynutil.add_weight((graph_hundred + hundredmillion_digit + graph_tenmillion), -1.9)
            | pynutil.add_weight((graph_hundred + hundredmillion_digit + pynutil.delete('0') + graph_million), -1.8)
            | pynutil.add_weight(
                (
                    graph_hundred
                    + hundredmillion_digit
                    + pynutil.delete('00')
                    + pynutil.insert('零')
                    + graph_hundredthousand
                ),
                -1.7,
            )
            | pynutil.add_weight(
                (
                    graph_hundred
                    + hundredmillion_digit
                    + (pynutil.delete('000') | (pynutil.delete('00,0') | pynutil.delete('00，0')))
                    + pynutil.insert('零')
                    + graph_tenthousand
                ),
                -1.6,
            )
            | pynutil.add_weight(
                (
                    graph_hundred
                    + hundredmillion_digit
                    + (pynutil.delete('0000') | (pynutil.delete('00,00') | pynutil.delete('00，00')))
                    + pynutil.insert('零')
                    + graph_thousand
                ),
                -1.5,
            )
            | pynutil.add_weight(
                (
                    graph_hundred
                    + hundredmillion_digit
                    + (pynutil.delete('00000') | (pynutil.delete('00,000,') | pynutil.delete('00，000，')))
                    + pynutil.insert('零')
                    + graph_hundred
                ),
                -1.4,
            )
            | pynutil.add_weight(
                (
                    graph_hundred
                    + hundredmillion_digit
                    + (pynutil.delete('000000') | (pynutil.delete('00,000,0') | pynutil.delete('00，000，0')))
                    + pynutil.insert('零')
                    + (graph_tens | graph_teen_alt)
                ),
                -1.3,
            )
            | pynutil.add_weight(
                (
                    graph_hundred
                    + hundredmillion_digit
                    + (pynutil.delete('0000000') | (pynutil.delete('00,000,00') | pynutil.delete('00，000，00')))
                    + pynutil.insert('零')
                    + digit
                ),
                -1.2,
            )
        )

        # 千亿; 100000000000-999999999999; hundred billions
        graph_hundredbillion = (
            pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + (pynutil.delete('00000000') | (pynutil.delete('00,000,000') | pynutil.delete('00，000，000')))
                ),
                -2.0,
            )
            | pynutil.add_weight((graph_thousand + hundredmillion_digit + graph_tenmillion), -1.9)
            | pynutil.add_weight((graph_thousand + hundredmillion_digit + pynutil.delete('0') + graph_million), -1.8)
            | pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + pynutil.delete('00')
                    + pynutil.insert('零')
                    + graph_hundredthousand
                ),
                -1.7,
            )
            | pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + (pynutil.delete('000') | (pynutil.delete('00,0') | pynutil.delete('00，0')))
                    + pynutil.insert('零')
                    + graph_tenthousand
                ),
                -1.6,
            )
            | pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + (pynutil.delete('0000') | (pynutil.delete('00,00') | pynutil.delete('00，00')))
                    + pynutil.insert('零')
                    + graph_thousand
                ),
                -1.5,
            )
            | pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + (pynutil.delete('00000') | (pynutil.delete('00,000,') | pynutil.delete('00，000，')))
                    + pynutil.insert('零')
                    + graph_hundred
                ),
                -1.4,
            )
            | pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + (pynutil.delete('000000') | (pynutil.delete('00,000,0') | pynutil.delete('00，000，0')))
                    + pynutil.insert('零')
                    + (graph_tens | graph_teen_alt)
                ),
                -1.3,
            )
            | pynutil.add_weight(
                (
                    graph_thousand
                    + hundredmillion_digit
                    + (pynutil.delete('0000000') | (pynutil.delete('00,000,00') | pynutil.delete('00，000，00')))
                    + pynutil.insert('零')
                    + digit
                ),
                -1.2,
            )
        )

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
        graph_mandarin = pynini.closure(
            (
                (
                    digit
                    | graph_teen
                    | graph_tens
                    | graph_hundred
                    | graph_thousand
                    | graph_tenthousand
                    | graph_hundredthousand
                )
                + suffix
            )
        )

        # combining all the graph above
        graph = pynini.union(
            pynutil.add_weight(graph_hundredbillion, -2.0),
            pynutil.add_weight(graph_tenbillion, -1.9),
            pynutil.add_weight(graph_billion, -1.8),
            pynutil.add_weight(graph_hundredmillion, -1.7),
            pynutil.add_weight(graph_tenmillion, -1.6),
            pynutil.add_weight(graph_million, -1.5),
            pynutil.add_weight(graph_hundredthousand, -1.4),
            pynutil.add_weight(graph_tenthousand, -1.3),
            pynutil.add_weight(graph_thousand, -1.2),
            pynutil.add_weight(graph_hundred, -1.1),
            pynutil.add_weight(graph_tens, -1.0),
            graph_teen,
            digit,
            zero,
        )

        # adding optional +(正)/-(负) signs
        graph_sign = (
            (pynutil.insert("positive: \"") + pynini.accep("正") + pynutil.insert("\""))
            | (pynutil.insert("negative: \"") + pynini.accep("负") + pynutil.insert("\""))
            | (pynutil.insert("negative: \"") + pynini.cross("負", "负") + pynutil.insert("\""))
            | (pynutil.insert("negative: \"") + pynini.cross("-", "负") + pynutil.insert("\""))
            | (pynutil.insert("positive: \"") + pynini.cross("+", "正") + pynutil.insert("\""))
        )

        graph_mandarin_sign = graph_sign + pynutil.insert(" ") + graph_mandarin
        # final graph
        final_graph_sign = (
            graph_sign + pynutil.insert(" ") + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        )
        final_graph_numbers_only = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        # imprted when building other grammars
        self.just_cardinals = graph | graph_mandarin | final_graph_sign | graph_mandarin_sign
        graph_mandarins = pynutil.insert("integer: \"") + graph_mandarin + pynutil.insert("\"")

        final_graph = final_graph_numbers_only | final_graph_sign | graph_mandarins | graph_mandarin_sign

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
