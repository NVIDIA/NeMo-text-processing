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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class MoneyFst(GraphFst):
    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        currency_unit = pynutil.delete('currency: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        number_unit = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        fraction_unit = pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        decimal_unit = (
            pynutil.insert(".")
            + pynutil.delete('fractional_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('quantity: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # regular money part
        graph_money_regular = (
            currency_unit + delete_space + number_unit + delete_space + pynutil.insert(".") + fraction_unit
        )
        graph_only_major_regular = currency_unit + delete_space + number_unit
        graph_only_minor_regular = currency_unit + delete_space + pynutil.insert("0.") + fraction_unit
        graph_large_money = currency_unit + delete_space + number_unit + delete_space + decimal_unit

        graph_regular = graph_money_regular | graph_only_major_regular | graph_only_minor_regular | graph_large_money

        major_symbol = pynini.accep("块")
        minor_symbol = pynini.accep("毛") | pynini.accep("角")
        lesser_symbol = pynini.accep("分")

        major_currency = pynutil.delete('currency_major: "') + major_symbol + pynutil.delete('"')
        minor_currency = pynutil.delete('currency_minor: "') + minor_symbol + pynutil.delete('"')
        lesser_currency = pynutil.delete('currency_min:"') + lesser_symbol + pynutil.delete('"')

        graph_kuai = number_unit + delete_space + major_currency
        graph_mao = number_unit + delete_space + minor_currency
        graph_fen = number_unit + delete_space + lesser_currency

        graph_kuaimao = graph_kuai + delete_space + fraction_unit + delete_space + minor_currency
        graph_kuaifen = graph_kuai + delete_space + fraction_unit + delete_space + lesser_currency
        graph_maofen = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('currency_minor: "')
            + minor_symbol
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('fraction_part: "')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('currency_min: "')
            + lesser_symbol
            + pynutil.delete('"')
        )

        graph_all = graph_kuai + delete_space + graph_maofen

        graph_mandarin = (
            (graph_kuai | graph_mao | graph_fen) | graph_kuaimao | graph_kuaifen | graph_maofen | graph_all
        )

        graph_verbalizer = graph_regular | pynutil.add_weight(graph_mandarin, -2.0)

        delete_tokens = self.delete_tokens(graph_verbalizer)
        self.fst = delete_tokens.optimize()
