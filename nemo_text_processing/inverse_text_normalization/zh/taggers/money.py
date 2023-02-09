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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil

class MoneyFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")

        # imports
        minor_currency_cent = pynini.string_file(get_abs_path("data/money/currency_rmb_minor_cent-nano.tsv"))
        minor_currency_tencent = pynini.string_file(get_abs_path("data/money/currency_rmb_minor_tencent-nano.tsv"))
        minor_digit = pynini.string_file(get_abs_path("data/numbers/digit-nano.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero-nano.tsv"))
        major_currency = pynini.string_file(get_abs_path("data/money/currency_major-nano.tsv"))  #
        minor_currency = pynini.string_file(get_abs_path("data/money/currency_minor-nano.tsv"))  #
        graph_cardinal = cardinal.for_ordinals
        graph_decimal = decimal.final_graph_wo_negative  #
        fraction_integer = minor_digit | zero

        # add leding zero to the number: 1 -> 01
        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)  #
        graph_fractional_values = graph_cardinal @ add_leading_zero_to_double_digit  #

        # regular number and yuan part
        graph_integer_component = pynutil.insert("integer_part: \"") + graph_cardinal + pynutil.insert("\"")
        graph_fractional_component = (
            pynutil.insert("fractional_part: \"")
            + graph_fractional_values
            + pynutil.insert("\"")
            + pynutil.delete(minor_currency)
        )
        graph_fractional_component_ex = (
            pynutil.insert("fractional_part: \"") + graph_fractional_values + pynutil.insert("\"")
        )

        # regular symbol part
        graph_major_currency = pynutil.insert("currency: \"") + major_currency + pynutil.insert("\"")
        graph_minor_currency = pynutil.insert("currency: \"") + minor_currency + pynutil.insert("\"")

        # regular combine number and symbol part
        graph_only_major = graph_integer_component + pynutil.insert(" ") + graph_major_currency
        graph_only_minor = graph_fractional_component_ex + pynutil.insert(" ") + graph_minor_currency
        graph_money = graph_only_major + pynutil.insert(" ") + graph_fractional_component

        # regular large money with decimals
        graph_large_money = graph_decimal + pynutil.insert(" ") + graph_major_currency

        # final graph for regular currency
        graph_regular_money = graph_only_major | graph_only_minor | graph_money | graph_large_money

        # yuan number part
        graph_cent_fractional_comp = pynutil.insert("cent_part: \"") + fraction_integer + pynutil.insert("\"")
        graph_tencent_fractional_comp = pynutil.insert("tencent_part: \"") + fraction_integer + pynutil.insert("\"")

        # yuan symbol part
        graph_currency_minor_cent = pynutil.insert("currency: \"") + minor_currency_cent + pynutil.insert("\"")
        graph_currency_minor_tencent = pynutil.insert("currency: \"") + minor_currency_tencent + pynutil.insert("\"")

        # yuan combine number and symbol part
        graph_only_cent = graph_cent_fractional_comp + pynutil.insert(" ") + graph_currency_minor_cent
        graph_only_tencent = graph_tencent_fractional_comp + pynutil.insert(" ") + graph_currency_minor_tencent

        # yuan major plus minor
        symbols = pynini.union('元', '毛', '角', '分')
        delete_symbols = pynutil.delete(symbols)
        graph_major_cent = (
            graph_integer_component
            + delete_symbols
            + pynutil.insert(" ")
            + graph_cent_fractional_comp
            + pynutil.insert(" ")
            + graph_currency_minor_cent
        )
        graph_major_tencent = (
            graph_integer_component
            + delete_symbols
            + pynutil.insert(" ")
            + graph_tencent_fractional_comp
            + pynutil.insert(" ")
            + graph_currency_minor_tencent
        )
        graph_tencent_cent = (
            graph_tencent_fractional_comp
            + delete_symbols
            + pynutil.insert(" ")
            + graph_cent_fractional_comp
            + pynutil.insert(" ")
            + graph_currency_minor_cent
        )
        graph_major_minor = (
            graph_integer_component
            + delete_symbols
            + pynutil.insert(" ")
            + graph_tencent_fractional_comp
            + pynutil.insert(" ")
            + delete_symbols
            + graph_cent_fractional_comp
            + pynutil.insert(" ")
            + graph_currency_minor_cent
        )

        # final graph for yuan
        graph_yuan_only = graph_only_cent | graph_only_tencent
        graph_yuan_comb = graph_major_cent | graph_major_tencent | graph_tencent_cent | graph_major_minor

        # combing both
        graph_yuan = graph_yuan_only | graph_yuan_comb
        graph_final = graph_regular_money | graph_yuan
        final = self.add_tokens(graph_final)
        self.fst = final.optimize()
