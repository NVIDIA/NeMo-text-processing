# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. बहत्तर लाइटकॉइन -> money { integer_part: "७२" currency: "ł" }
        e.g. बहत्तर मोनेरो -> money { integer_part: "७२" currency: "ɱ" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        money: MoneyFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        cardinal_single_and_double_digit_graph = cardinal.graph_digit | cardinal.graph_teens_and_ties
        decimal_graph = decimal.final_graph_wo_negative
        currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.integer_quarterly_measures = pynutil.insert("integer_part: \"") + cardinal_single_and_double_digit_graph
        self.integer_paune = pynutil.insert("integer_part: \"") + paune_graph
        self.paise = pynutil.insert("fractional_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.fraction = decimal_graph
        self.currency = pynutil.insert("currency: \"") + currency_graph + pynutil.insert("\" ")
        aur = pynutil.delete("और")
        delete_hundred = pynutil.delete("सौ")
        delete_lakh = pynutil.delete("लाख")
        delete_hazar = pynutil.delete("हजार") | pynutil.delete("हज़ार")
        delete_crore = pynutil.delete("करोड़") | pynutil.delete("करोड़")

        graph_currency_decimal = self.fraction + delete_extra_space + self.currency
        graph_currency_cardinal = self.integer + delete_extra_space + self.currency

        graph_rupay_and_paisa = (
            graph_currency_cardinal
            + delete_extra_space
            + pynini.closure(aur + delete_extra_space, 0, 1)
            + self.paise
            + delete_extra_space
            + pynutil.delete(currency_graph)
        )
        
        #cases for saade,sava,paune,dedh and dhaai with hundreds and thousands
        graph_exceptions = self.integer + delete_extra_space + self.currency
        
        #exceptions with lakhs
        graph_saade_lakh = pynutil.add_weight(pynutil.delete("साढ़े") + delete_space + self.integer_quarterly_measures + delete_space + pynutil.insert("५००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_lakh + delete_extra_space + self.currency, 0.01)
        graph_sava_lakh = pynutil.add_weight(pynutil.delete("सवा") + delete_space + self.integer_quarterly_measures + delete_space + pynutil.insert("२५०००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_lakh + delete_extra_space + self.currency, 0.01)
        graph_paune_lakh = pynutil.delete("पौने") + delete_space + self.integer_paune + delete_space + pynutil.insert("७५०००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_lakh + delete_extra_space + self.currency
        graph_dedh_lakh = pynutil.delete("डेढ़") + delete_space + pynutil.insert("integer_part: \"") + pynutil.insert("१५००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_lakh + delete_extra_space + self.currency
        graph_dhaai_lakh = pynutil.delete("ढाई") + delete_space + pynutil.insert("integer_part: \"") + pynutil.insert("२५००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_lakh + delete_extra_space + self.currency
        graph_exceptions_lakhs = graph_saade_lakh | graph_sava_lakh | graph_paune_lakh | graph_dedh_lakh | graph_dhaai_lakh
        
        # exceptions with crores
        graph_saade_crore = pynutil.delete("साढ़े") + delete_space + self.integer_quarterly_measures + delete_space + pynutil.insert("५००००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_crore + delete_extra_space + self.currency
        graph_sava_crore = pynutil.delete("सवा") + delete_space + self.integer_quarterly_measures + delete_space + pynutil.insert("२५०००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_crore + delete_extra_space + self.currency
        graph_paune_crore = pynutil.delete("पौने") + delete_space + self.integer_paune + delete_space + pynutil.insert("७५०००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_crore + delete_extra_space + self.currency
        graph_dhaai_crore = pynutil.delete("ढाई") + delete_space + pynutil.insert("integer_part: \"") + pynutil.insert("२५००००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_crore + delete_extra_space + self.currency
        graph_dedh_crore = pynutil.delete("डेढ़") + delete_space + pynutil.insert("integer_part: \"") + pynutil.insert("१५००००००", weight=-0.1) + pynutil.insert("\"") + delete_space + delete_crore + delete_extra_space + self.currency
        graph_exceptions_crores = graph_saade_crore | graph_sava_crore | graph_paune_crore | graph_dedh_crore | graph_dhaai_crore
        
        
        graph_quarterly_measures = graph_exceptions | graph_exceptions_lakhs | graph_exceptions_crores
        
        graph = graph_currency_decimal | graph_currency_cardinal | graph_rupay_and_paisa | graph_quarterly_measures
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
