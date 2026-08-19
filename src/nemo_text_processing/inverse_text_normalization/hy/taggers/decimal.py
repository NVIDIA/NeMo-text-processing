# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hy.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    delete_space,
)


def get_quantity(
    decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike', input_case: str = INPUT_LOWER_CASED
) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. հինգ միլիոն -> tokens { decimal { integer_part: "5" quantity: "միլիոն" } }
    e.g. հինգ ամբողջ յոթ միլիարդ -> tokens { decimal { integer_part: "5"  fractional_part: "7" quantity: "միլիարդ" } }

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
        input_case: accepting either "lower_cased" or "cased" input.
        (input_case is not necessary everything is made for lower_cased input)
        TODO add case input support

    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
    )

    suffix = pynini.union("միլիոն", "միլիարդ", "տրիլիոն")

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= decimal + delete_extra_space + pynutil.insert("quantity: \"") + (suffix | "հազար") + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. հիսուն ու կես տրիլիոն -> decimal { integer_part: "50"  fractional_part: "5" quantity: "տրիլիոն" }
        e.g. մեկ միլիարդ -> decimal { integer_part: "1" quantity: "միլիարդ" }
    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
        TODO add cased input support
    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) | pynini.string_map(
            [("զրո", "0"), ("կես", "5")]
        )

        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.only_decimal = graph_decimal.optimize()

        point_first = pynutil.delete("ամբողջ")
        point_second = pynutil.delete("ու")

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure((graph_integer | pynini.string_map(["", "0"])) + delete_extra_space, 0, 1)
            + (point_first | point_second)
            + delete_extra_space
            + graph_fractional
        )
        final_graph = final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal.graph_hundred_component_at_least_one_none_zero_digit, input_case=input_case
        )

        self.final_graph_wo_negative |= pynutil.add_weight(
            pynini.compose(TO_LOWER + NEMO_SIGMA, self.final_graph_wo_negative).optimize(), MIN_NEG_WEIGHT
        )

        quantity_graph = get_quantity(
            final_graph_wo_sign, cardinal.graph_hundred_component_at_least_one_none_zero_digit, input_case=input_case
        )
        final_graph |= quantity_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
