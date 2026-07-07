# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.el.graph_utils import shift_cardinal_gender_fem
from nemo_text_processing.text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, insert_space

# Each currency: symbols/codes that denote it, its grammatical gender, the singular and plural
# major-unit nouns, and the singular and plural minor-unit (cents) nouns. The amount agrees in
# gender with the currency, so e.g. λίρα (feminine) yields "τρεις λίρες" not "τρία λίρες".
_CURRENCIES = [
    (["€", "EUR", "ευρώ"], "n", "ευρώ", "ευρώ", "λεπτό", "λεπτά"),
    (["$", "USD"], "n", "δολάριο", "δολάρια", "σεντ", "σεντ"),
    (["£", "GBP"], "f", "λίρα", "λίρες", "πένα", "πένες"),
]


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money in Greek, e.g.
        €5 -> money { integer_part: "πέντε ευρώ" }
        €3,50 -> money { integer_part: "τρία ευρώ" fractional_part: "πενήντα λεπτά" }
        £1 -> money { integer_part: "μία λίρα" }

    The currency symbol may precede or follow the amount. The amount agrees in gender with the
    currency and the noun is singular when the amount is one.

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst (unused for now, kept for interface symmetry)
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        base_amount = cardinal.graph_no_tokens
        amount_no_one = pynini.difference(pynini.project(base_amount, "input"), pynini.accep("1")) @ base_amount

        # cents amount: exactly two digits, 01-99, neuter
        cents_one = pynini.cross("01", "ένα")
        cents_multi = (pynutil.delete("0") + graph_digit) | cardinal.graph_tens
        cents_multi = pynini.difference(pynini.project(cents_multi, "input"), pynini.accep("01")) @ cents_multi

        money_graph = None
        for symbols, gender, major_sg, major_pl, minor_sg, minor_pl in _CURRENCIES:
            symbol = pynini.union(*symbols)
            if gender == "f":
                amount_multi = shift_cardinal_gender_fem(amount_no_one)
                one_word = "μία"
            else:
                amount_multi = amount_no_one
                one_word = "ένα"

            major_one = pynini.cross("1", one_word) + insert_space + pynutil.insert(major_sg)
            major_many = amount_multi + insert_space + pynutil.insert(major_pl)
            major = major_one | major_many

            cent_one = cents_one + insert_space + pynutil.insert(minor_sg)
            cent_many = cents_multi + insert_space + pynutil.insert(minor_pl)
            cents = cent_one | cent_many

            integer_field = pynutil.insert("integer_part: \"") + major + pynutil.insert("\"")
            cents_field = pynutil.insert(" fractional_part: \"") + cents + pynutil.insert("\"")
            amount_with_cents = integer_field + pynutil.delete(",") + cents_field
            amount = amount_with_cents | integer_field

            # symbol/code before the amount (with optional space) or after it (with optional space)
            delete_opt_space = pynini.closure(pynutil.delete(" "), 0, 1)
            before = pynutil.delete(symbol) + delete_opt_space + amount
            after = amount + delete_opt_space + pynutil.delete(symbol)
            currency_graph = before | after

            money_graph = currency_graph if money_graph is None else (money_graph | currency_graph)

        self.final_graph = money_graph
        self.fst = self.add_tokens(money_graph).optimize()
