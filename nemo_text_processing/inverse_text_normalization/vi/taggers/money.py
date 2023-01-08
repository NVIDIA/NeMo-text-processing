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
from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    convert_space,
    delete_extra_space,
)
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. mười hai đô la mỹ -> money { integer_part: "12" currency: "$" }
        e.g. mười phẩy chín đồng -> money { integer_part: "10.9" currency: "đ" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception
        graph_decimal_final = decimal.final_graph_wo_negative
        graph_half = pynini.cross("rưỡi", "5")

        unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit_singular = pynini.invert(unit)

        graph_unit_singular = pynutil.insert('currency: "') + convert_space(unit_singular) + pynutil.insert('"')

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)

        # twelve dollars fifty, only after integer
        optional_cents_suffix = pynini.closure(
            delete_extra_space
            + pynutil.insert('fractional_part: "')
            + (pynutil.add_weight(cardinal_graph @ add_leading_zero_to_double_digit, -0.7) | graph_half)
            + pynutil.insert('"'),
            0,
            1,
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
            + delete_extra_space
            + graph_unit_singular
            + optional_cents_suffix
        )

        graph_decimal = graph_decimal_final + delete_extra_space + graph_unit_singular + optional_cents_suffix
        final_graph = graph_integer | graph_decimal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
