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
from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        currency_unit = pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        number_unit = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        fraction_unit = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )
        cent_unit = pynutil.delete("cent_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        tencent_unit = pynutil.delete("tencent_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        decimal_unit = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("quantity: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # regular money part
        graph_money_regular = (
            currency_unit + delete_space + number_unit + delete_space + pynutil.insert(".") + fraction_unit
        )
        graph_only_major_regular = currency_unit + delete_space + number_unit
        graph_only_minor_regular = currency_unit + delete_space + pynutil.insert("0.") + fraction_unit
        graph_large_money = currency_unit + delete_space + number_unit + delete_space + decimal_unit

        graph_regular = graph_money_regular | graph_only_major_regular | graph_only_minor_regular | graph_large_money

        # yuan part
        graph_money_yuan = (
            currency_unit
            + delete_space
            + number_unit
            + delete_space
            + pynutil.insert(".")
            + ((pynutil.insert("0") + cent_unit) | (tencent_unit) | (tencent_unit + delete_space + cent_unit))
        )
        graph_yuan_minors = (
            currency_unit + delete_space + pynutil.insert("0.") + tencent_unit + delete_space + cent_unit
        )
        graph_yuan = graph_money_yuan | graph_yuan_minors

        graph_verbalizer = graph_regular | graph_yuan

        delete_tokens = self.delete_tokens(graph_verbalizer)
        self.fst = delete_tokens.optimize()
