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

from nemo_text_processing.inverse_text_normalization.mr.graph_utils import (
    MINUS,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.mr.utils import get_abs_path, load_labels


def get_quantity(decimal, cardinal_fst):
    numbers = cardinal_fst @ (
        pynutil.delete(pynini.closure("०")) + pynini.difference(NEMO_DIGIT, "०") + pynini.closure(NEMO_DIGIT)
    )
    suffix_labels = load_labels(get_abs_path("/data/numbers/thousands.tsv"))
    suffix_labels = [x[0] for x in suffix_labels if x[0] != "हजार"]
    suffix = pynini.union(*suffix_labels).optimize()

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= decimal + delete_extra_space + pynutil.insert("quantity: \"") + (suffix | "हजार") + pynutil.insert("\"")

    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. तेहतीस पूर्णांक तीन -> decimal { integer_part: "३३" fractional_part: "३" }
        e.g. उणे तेहतीस पूर्णांक तीन लाख -> decimal { negative: "true" integer_part: "३३" fractional_part: "३" quantity: "लाख" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digits = pynini.string_file(get_abs_path("data/numbers/digits.tsv")).invert()
        decimal_word = pynini.cross("पूर्णांक", "")
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"true\"") + delete_extra_space, 0, 1,
        )
        graph_integer = (
            pynutil.insert("integer_part: \"")
            + pynini.closure(cardinal.graph, 0, 1)
            + pynutil.insert("\"")
            + NEMO_SPACE
        )
        graph_decimal = graph_integer + delete_space + decimal_word

        graph_fractional = (
            pynutil.insert("fractional_part: \"")
            + pynini.closure(delete_space + (graph_zero | graph_digits), 1)
            + pynutil.insert("\"")
        )
        graph_decimal += graph_fractional

        final_graph_without_sign = graph_decimal
        final_graph = optional_graph_negative + final_graph_without_sign

        self.final_graph_without_negative = final_graph_without_sign | get_quantity(
            final_graph_without_sign, cardinal.graph_hundred_component_at_least_one_non_zero_digit
        )

        quantity_graph = get_quantity(
            final_graph_without_sign, cardinal.graph_hundred_component_at_least_one_non_zero_digit
        )
        final_graph |= optional_graph_negative + quantity_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
