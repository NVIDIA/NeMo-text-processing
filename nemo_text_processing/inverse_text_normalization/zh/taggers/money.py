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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path


class MoneyFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")

        # imports
        major_currency = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        minor_currency = pynini.string_file(get_abs_path("data/money/currency_minor.tsv"))
        digits = pynini.string_file(get_abs_path("data/numbers/digit-nano.tsv"))
        graph_cardinal = cardinal.for_ordinals
        graph_decimal = decimal.final_graph_wo_negative  #

        # add leding zero to the number: 1 -> 01
        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)  #
        graph_fractional_values = graph_cardinal @ add_leading_zero_to_double_digit  #

        # regular number and yuan part
        graph_integer_component = pynutil.insert('integer_part: "') + graph_cardinal + pynutil.insert('"')
        graph_fractional_component = (
            pynutil.insert('fractional_part: "') + graph_fractional_values + pynutil.insert('"')
        )

        # regular symbol part
        graph_major_currency = pynutil.insert('currency: "') + major_currency + pynutil.insert('"')
        graph_minor_currency = pynutil.insert('currency: "') + minor_currency + pynutil.insert('"')

        # regular combine number and symbol part
        graph_only_major = graph_integer_component + pynutil.insert(" ") + graph_major_currency
        graph_only_minor = graph_fractional_component + pynutil.insert(" ") + graph_minor_currency
        graph_money = graph_only_major + pynutil.insert(" ") + graph_fractional_component

        # regular large money with decimals
        graph_large_money = graph_decimal + pynutil.insert(" ") + graph_major_currency

        # final graph for regular currency
        graph_regular_money = graph_only_major | graph_only_minor | graph_money | graph_large_money

        # yuan major plus minor
        major_symbol = pynini.accep("块") | pynini.cross("塊", "块")
        tencent = pynini.accep("毛") | pynini.accep("角",)
        cent = pynini.accep("分")
        graph_kuai = (
            graph_integer_component
            + pynutil.insert(" ")
            + pynutil.insert('currency_major: "')
            + pynini.closure(major_symbol, 1, 1)
            + pynutil.insert('"')
        )
        graph_mao = (
            graph_integer_component
            + pynutil.insert(" ")
            + pynutil.insert('currency_minor: "')
            + pynini.closure(tencent, 1, 1)
            + pynutil.insert('"')
        )
        graph_fen = (
            graph_integer_component
            + pynutil.insert(" ")
            + pynutil.insert('currency_minor: "')
            + pynini.closure(cent, 1, 1)
            + pynutil.insert('"')
        )

        graph_digits = pynutil.insert('fractional_part: "') + digits + pynutil.insert('"')
        graph_kuaimao = (
            graph_kuai
            + pynutil.insert(" ")
            + graph_digits
            + pynutil.insert(" ")
            + pynutil.insert('currency_minor: "')
            + pynini.closure(tencent, 1, 1)
            + pynutil.insert('"')
        )
        graph_kuaifen = (
            graph_kuai
            + pynutil.insert(" ")
            + graph_digits
            + pynutil.insert(" ")
            + pynutil.insert('currency_minor: "')
            + pynini.closure(cent, 1, 1)
            + pynutil.insert('"')
        )
        graph_maofen = (
            pynutil.insert('fractional_part: "')
            + digits
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + pynutil.insert('currency_minor: "')
            + pynini.closure(tencent, 1, 1)
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + pynutil.insert('fraction_part: "')
            + digits
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + pynutil.insert('currency_min: "')
            + pynini.closure(cent, 1, 1)
            + pynutil.insert('"')
        )

        graph_kuaimaofen = (
            graph_kuai
            + pynutil.insert(" ")
            + pynutil.insert('fractional_part: "')
            + digits
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + pynutil.insert('currency_minor: "')
            + pynini.closure(tencent, 1, 1)
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + pynutil.insert('fraction_part: "')
            + digits
            + pynutil.insert('"')
            + pynutil.insert(" ")
            + pynutil.insert('currency_min: "')
            + pynini.closure(cent, 1, 1)
            + pynutil.insert('"')
        )

        graph_mandarin = (
            graph_kuai | graph_mao | graph_fen | graph_kuaimao | graph_kuaifen | graph_maofen | graph_kuaimaofen
        )

        # combing both
        graph_final = graph_regular_money | graph_mandarin
        final = self.add_tokens(graph_final)
        self.fst = final.optimize()
