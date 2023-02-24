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
from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    capitalized_input_graph,
    convert_space,
    delete_extra_space,
    delete_space,
    get_singulars,
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
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception
        # add support for missing hundred (only for 3 digit numbers)
        # "one fifty" -> "one hundred fifty"
        with_hundred = pynini.compose(
            pynini.closure(NEMO_NOT_SPACE) + pynini.accep(" ") + pynutil.insert("hundred ") + NEMO_SIGMA,
            pynini.compose(cardinal_graph, NEMO_DIGIT ** 3),
        )
        cardinal_graph |= with_hundred
        graph_decimal_final = decimal.final_graph_wo_negative
        unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit_singular = pynini.invert(unit)

        if input_case == INPUT_CASED:
            unit_singular = capitalized_input_graph(unit_singular)
        unit_plural = get_singulars(unit_singular)

        graph_unit_singular = pynutil.insert("currency: \"") + convert_space(unit_singular) + pynutil.insert("\"")
        graph_unit_plural = (
            pynutil.insert("currency: \"") + convert_space(unit_plural | unit_singular) + pynutil.insert("\"")
        )

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)

        one_graph = pynini.accep("one").optimize()
        if input_case == INPUT_CASED:
            one_graph |= pynini.accep("One").optimize()

        cent_graph = pynutil.delete("cent")
        cents_graph = pynutil.delete("cents")

        if input_case == INPUT_CASED:
            cent_graph |= pynutil.delete("Cent")
            cents_graph |= pynutil.delete("Cents")

        # twelve dollars (and) fifty cents, zero cents
        cents_standalone = (
            pynutil.insert("fractional_part: \"")
            + pynini.union(
                pynutil.add_weight(((NEMO_SIGMA - one_graph) @ cardinal_graph), -0.7)
                @ add_leading_zero_to_double_digit
                + delete_space
                + cents_graph,
                pynini.cross(one_graph, "01") + delete_space + cent_graph,
            )
            + pynutil.insert("\"")
        )

        optional_cents_standalone = pynini.closure(
            delete_space
            + pynini.closure(pynutil.delete("and") + delete_space, 0, 1)
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
            + graph_unit_plural
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_integer |= (
            pynutil.insert("integer_part: \"")
            + pynini.cross(one_graph, "1")
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_unit_singular
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_decimal = graph_decimal_final + delete_extra_space + graph_unit_plural
        graph_decimal |= pynutil.insert("currency: \"$\" integer_part: \"0\" ") + cents_standalone
        final_graph = graph_integer | graph_decimal

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
