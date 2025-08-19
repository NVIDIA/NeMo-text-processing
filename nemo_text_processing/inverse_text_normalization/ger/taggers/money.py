# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    NEMO_SPACE,
    NEMO_DIGIT,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money.
    The grammar assumes centesimal currency design, whereby the main unit of currency is divided into 100 subunits.
    While this assumption will cover the vast majority of world currencies, there will be exceptions such as e.g. Mauritanian Ouguiya or Malagasy Ariary.
        e.g. sieben euro -> money { integer_part: "7" currency: "€" }
        e.g. ein million dollar -> { integer_part: "1.000.000" currency: "$" }
        e.g. sechshunderteinundzwanzig yen und achtzehn sen -> money { integer_part: "621" currency: "¥" fractional_part: "18" }
        e.g. zehn pence -> money { integer_part: "0" fractional_part: "10" currency: "£" }

    """

    def __init__(
        self,
        cardinal: GraphFst,
    ):
        super().__init__(name="money", kind="classify")
        morphemes_for_one = (
            pynini.accep("ein") + pynini.accep("e").ques + pynini.union(*"rnsm").ques
        )
        declensions_of_one = pynini.cross(morphemes_for_one, "1")
        graph_cardinal = cardinal.graph_all_cardinals | declensions_of_one
        graph_centiles = cardinal.graph_single_and_double_digits
        currency_major = pynini.string_file(
            get_abs_path("data/money/currency_major.tsv")
        )
        currency_minor = pynini.string_file(
            get_abs_path("data/money/currency_minor.tsv")
        )
        currency_minor_to_major = pynini.string_file(
            get_abs_path("data/money/currency_minor_to_major.tsv")
        )

        graph_currency_major = (
            pynutil.insert('currency: "') + currency_major + pynutil.insert('"')
        )
        graph_currency_min_to_maj = (
            pynutil.insert('currency: "')
            + (currency_minor_to_major @ currency_major)
            + pynutil.insert('"')
        )
        cents = pynini.accep("cent") | pynini.accep("cents")
        graph_cents = pynini.cross(cents, "¢")
        graph_currency_cents = (
            pynutil.insert('currency: "') + graph_cents + pynutil.insert('"')
        )

        # zehn Euro -> 10 €
        graph_currency_major_only = (
            pynutil.insert('integer_part: "')
            + graph_cardinal
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + graph_currency_major
        )

        # zehn Cent -> 10 ¢
        # this subgraph accounts for the contexts where the major currency
        # that the centiles belong to is unspecified
        graph_currency_cents_only = (
            pynutil.insert('integer_part: "0"')
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('fractional_part: "')
            + (
                graph_centiles
                | pynini.cross("hundert", "100")
                | pynini.cross("einhundert", "100")
                | pynini.cross("ein hundert", "100")
            )
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + graph_currency_cents
        )

        # zehn Eurocent -> 0,10 €
        # the subgraph maps the centile minor currencies to their major counterparts
        # the abbreviations for minor currencies (e.g. Ø for the Danish øre) are not represented in this class
        graph_currency_minor_to_major = (
            pynutil.insert('integer_part: "0"')
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('fractional_part: "')
            + graph_centiles
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + graph_currency_min_to_maj
        )

        # zehn Euro und zehn Cent -> 10,10 €
        graph_currency_major_fractional = (
            graph_currency_major_only
            + (pynutil.delete(NEMO_SPACE) + pynutil.delete("und")).ques
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert('fractional_part: "')
            + graph_centiles
            + pynutil.insert('"')
            + (
                pynutil.delete(NEMO_SPACE)
                + pynutil.delete(pynini.project(currency_minor, "input"))
            ).ques
        )

        # Handles major currencies expressed as decimal fractions
        graph_currency_decimal = (
            pynutil.insert('integer_part: "')
            + graph_cardinal
            + pynutil.insert('"')
            + (
                pynini.cross(" komma ", NEMO_SPACE)
                | pynini.cross(" Komma ", NEMO_SPACE)
            )
            + pynutil.insert('fractional_part: "')
            + (graph_centiles | pynini.closure(NEMO_DIGIT, 1))
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            
            + graph_currency_major
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
        self.fst = final_graph
