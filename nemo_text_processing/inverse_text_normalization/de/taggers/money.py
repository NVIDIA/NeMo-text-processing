# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money.
    The grammar assumes centesimal currency design, whereby the main unit of currency is divided into 100 subunits.
    While this assumption will cover the vast majority of world currencies, there will be exceptions such as e.g. Mauritanian Ouguiya or Malagasy Ariary.
        e.g. sieben euro -> money { integer_part: "7" currency: "€" }
        e.g. eine million dollar -> money { integer_part: "1" quantity: "Million" currency: "$" }
        e.g. sechshunderteinundzwanzig yen und achtzehn sen -> money { integer_part: "621" currency: "¥" fractional_part: "18" }
        e.g. zehn pence -> money { integer_part: "0" fractional_part: "10" currency: "£" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
    ):
        super().__init__(name="money", kind="classify")

        graph_cardinal = cardinal.graph_no_exception
        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        graph_centiles = (graph_cardinal @ add_leading_zero_to_double_digit).optimize()
        graph_decimal_final = decimal.final_graph_wo_negative
        currency_major = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        currency_minor = pynini.string_file(get_abs_path("data/money/currency_minor.tsv"))
        currency_minor_to_major = pynini.string_file(get_abs_path("data/money/currency_minor_to_major.tsv"))

        graph_currency_major = pynutil.insert('currency: "') + currency_major + pynutil.insert('"')
        graph_currency_min_to_maj = (
            pynutil.insert('currency: "') + (currency_minor_to_major @ currency_major) + pynutil.insert('"')
        )
        cents = pynini.accep("cent") | pynini.accep("cents")

        # zehn Euro -> 10 €
        graph_currency_major_only = (
            pynutil.insert('integer_part: "')
            + graph_cardinal
            + pynutil.insert('"')
            + delete_extra_space
            + graph_currency_major
        )

        # zehn Cent -> 0,10 €
        graph_currency_cents_only = (
            pynutil.insert('integer_part: "0"')
            + insert_space
            + pynutil.insert('fractional_part: "')
            + graph_centiles
            + pynutil.insert('"')
            + delete_space
            + pynutil.delete(cents)
            + insert_space
            + pynutil.insert('currency: "€"')
        )

        # zehn Eurocent -> 0,10 €
        # the subgraph maps the centile minor currencies to their major counterparts
        # the abbreviations for minor currencies (e.g. Ø for the Danish øre) are not represented in this class
        graph_currency_minor_to_major = (
            pynutil.insert('integer_part: "0"')
            + insert_space
            + pynutil.insert('fractional_part: "')
            + graph_centiles
            + pynutil.insert('"')
            + delete_extra_space
            + graph_currency_min_to_maj
        )

        # zehn Euro und zehn Cent -> 10,10 €
        cents_part = pynutil.insert('fractional_part: "') + graph_centiles + pynutil.insert('"')
        minor_word = delete_space + pynutil.delete(pynini.project(currency_minor, "input"))
        graph_currency_major_fractional = graph_currency_major_only + (
            (delete_space + pynutil.delete("und") + delete_extra_space + cents_part + minor_word)
            | (delete_extra_space + cents_part + minor_word.ques)
        )

        # zwei Millionen Euro -> 2 Millionen €
        graph_currency_decimal = pynutil.add_weight(
            graph_decimal_final + delete_extra_space + graph_currency_major, -0.1
        )

        graph_all_currency = (
            graph_currency_major_only
            | graph_currency_cents_only
            | graph_currency_major_fractional
            | graph_currency_minor_to_major
            | graph_currency_decimal
        )

        currency_graph = graph_all_currency.optimize()
        final_graph = self.add_tokens(currency_graph)
        self.fst = final_graph.optimize()
