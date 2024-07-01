# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    delete_optional_and,
    insert_space,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. twelve dollars and five cents -> money { integer_part: "12" fractional_part: 05 currency: "$" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception

        graph_decimal_final = decimal.final_graph_wo_negative
        unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit = pynini.invert(unit)
        graph_unit = pynutil.insert("currency: \"") + convert_space(unit) + pynutil.insert("\"")

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)

        one_graph = pynini.accep("אחד").optimize()
        cents_graph = pynutil.delete("אגורות")

        # twelve dollars (and) fifty cents, zero cents
        cents_standalone = (
                pynutil.insert("fractional_part: \"")
                + delete_optional_and
                + pynini.union(
                pynutil.add_weight(((NEMO_SIGMA - one_graph) @ cardinal_graph), -0.7)
                @ add_leading_zero_to_double_digit
                + delete_space
                + cents_graph,
                pynini.cross("אגורה", "01"),
            )
                + pynutil.insert("\"")
        )

        optional_cents_standalone = pynini.closure(
            delete_space
            + delete_optional_and
            + insert_space
            + cents_standalone,
            0,
            1,
        )

        # twelve dollars fifty, only after integer
        optional_cents_suffix = pynini.closure(
            delete_extra_space
            + pynutil.insert("fractional_part: \"")
            + pynutil.add_weight(cardinal_graph @ add_leading_zero_to_double_digit, -0.7)
            + pynutil.insert("\""),
            0,
            1,
        )

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + ((NEMO_SIGMA - one_graph) @ cardinal_graph)
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_unit
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_integer |= (
            pynutil.insert("integer_part: \"")
            + pynini.cross("שקל", "1")
            + pynutil.insert("\" currency: \"₪\"")
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_decimal = graph_decimal_final + delete_extra_space + graph_unit
        graph_decimal |= pynutil.insert("currency: \"₪\" integer_part: \"0\" ") + cents_standalone
        final_graph = graph_integer | graph_decimal

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
    from nemo_text_processing.inverse_text_normalization.he.taggers.decimal import DecimalFst

    cardinal = CardinalFst()
    decimal = DecimalFst(cardinal)
    graph = MoneyFst(cardinal, decimal).fst
    apply_fst("שלושה שקלים", graph)
    apply_fst("שלושה ש״ח", graph)
    apply_fst("שלושה שקלים חדשים", graph)
    apply_fst("שלושה שקלים וחמישים אגורות", graph)
    apply_fst("שני שקלים ותשעים ותשע אגורות", graph)
    apply_fst("חמש אגורות", graph)
    apply_fst("ארבעים ושבע יורו", graph)
    apply_fst("ארבעים ושבע דולרים", graph)
    apply_fst("שקל", graph)
    apply_fst("שקל ותשעים", graph)
    # apply_fst("שקל וחצי", graph)  # TODO

