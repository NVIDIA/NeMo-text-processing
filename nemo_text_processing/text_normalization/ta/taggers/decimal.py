# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_text_processing.text_normalization.ta.graph_utils import (
    COMMA,
    MINUS,
    NEMO_ALL_DIGIT,
    NEMO_DIGIT,
    PERIOD,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals
        e.g. "1.5" -> decimal { integer_part: "ஒன்று" fractional_part: "புள்ளி ஐந்து" }
        e.g. "-2.67" -> decimal { negative: "true" integer_part: "இரண்டு" fractional_part: "புள்ளி ஆறு ஏழு" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        ta_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        delete_point = pynutil.delete(PERIOD)
        zeros = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        sign = pynini.closure(pynini.accep(MINUS), 0, 1)

        optional_sign = pynini.closure(pynutil.insert("negative: ") + pynini.cross(MINUS, '"true" '), 0, 1)

        def field(name, body):
            return pynutil.insert(f'{name}: "') + body + pynutil.insert('"')

        def build(digits):
            zero = pynini.compose(pynini.project(zeros, "input"), digits).optimize()
            nonzero = pynini.difference(digits, zero).optimize()
            valid_int = (zero | nonzero + pynini.closure(digits | pynini.accep(COMMA))).optimize()

            integer = pynini.compose(valid_int, cardinal.final_graph).optimize()
            fraction = pynini.compose(pynini.closure(digits, 1), cardinal.single_digits_graph).optimize()
            leading_zero = pynini.compose(zero + pynini.closure(digits, 1), cardinal.single_digits_graph).optimize()

            with_int = (
                field("integer_part", integer) + delete_point + insert_space + field("fractional_part", fraction)
            )
            without_int = delete_point + field("fractional_part", fraction)
            leading_zero_decimal = (
                field("integer_part", leading_zero)
                + delete_point
                + insert_space
                + pynutil.insert('fractional_part: "' + PERIOD + ' ')
                + fraction
                + pynutil.insert('"')
            )
            return (leading_zero_decimal | with_int | without_int).optimize()

        final_graph = (optional_sign + (build(NEMO_DIGIT) | build(ta_digit))).optimize()

        def script_decimal(digit):
            return pynini.closure(digit | pynini.accep(COMMA), 0) + PERIOD + pynini.closure(digit, 1)

        mixed = pynini.difference(
            script_decimal(NEMO_ALL_DIGIT), script_decimal(NEMO_DIGIT) | script_decimal(ta_digit)
        ).optimize()
        mixed = field("name", sign + mixed).optimize()

        self.fst = (self.add_tokens(final_graph) | mixed).optimize()
