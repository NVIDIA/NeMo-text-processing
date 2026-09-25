# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    COMMA, MINUS, NEMO_ALL_DIGIT, NEMO_DIGIT, PERIOD, GraphFst, insert_space,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals
        e.g. "1.5" -> decimal { integer_part: "ஒன்று" fractional_part: "புள்ளி ஐந்து" }
        e.g. "-2.67" -> decimal { negative: "true" integer_part: "இரண்டு" fractional_part: "புள்ளி ஆறு ஏழு" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)
        ta_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        delete_point = pynutil.delete(PERIOD)
        optional_sign = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, '"true" '), 0, 1
        ).optimize()

        zeros = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        def same_script(digits):
            zero = pynini.compose(pynini.project(zeros, "input"), digits).optimize()
            nonzero = pynini.difference(digits, zero)
            valid_int = (zero | (nonzero + pynini.closure(digits | pynini.accep(COMMA)))).optimize()
            integer = pynini.compose(valid_int, cardinal.final_graph).optimize()
            fraction = pynini.compose(pynini.closure(digits, 1), cardinal.single_digits_graph).optimize()

            with_int = (
                pynutil.insert('integer_part: "') + integer + pynutil.insert('"') + delete_point + insert_space
                + pynutil.insert('fractional_part: "') + fraction + pynutil.insert('"')
            ).optimize()
            without_int = (
                pynutil.insert('has_integer: "false" ') + delete_point
                + pynutil.insert('fractional_part: "') + fraction + pynutil.insert('"')
            ).optimize()

            leading_zero_int_input = zero + pynini.closure(digits, 1)
            leading_zero_int = pynini.compose(leading_zero_int_input, cardinal.single_digits_graph).optimize()
            literal_point = (
                pynutil.insert('integer_part: "') + leading_zero_int + pynutil.insert('" ')
                + pynutil.insert('literal_point: "true" ')
                + delete_point
                + pynutil.insert('fractional_part: "') + fraction + pynutil.insert('"')
            ).optimize()

            return (with_int | without_int | literal_point).optimize()

        final_graph = (optional_sign + (same_script(NEMO_DIGIT) | same_script(ta_digit))).optimize()

        # Mixed-script decimals
        pure_ascii = pynini.closure(NEMO_DIGIT | pynini.accep(COMMA), 0) + PERIOD + pynini.closure(NEMO_DIGIT, 1)
        pure_tamil = pynini.closure(ta_digit | pynini.accep(COMMA), 0) + PERIOD + pynini.closure(ta_digit, 1)
        mixed_int = pynini.closure(NEMO_ALL_DIGIT | pynini.accep(COMMA), 0) + PERIOD + pynini.closure(NEMO_ALL_DIGIT, 1)
        mixed_pattern = pynini.difference(mixed_int, pure_ascii | pure_tamil).optimize()
        mixed = (
            pynutil.insert('name: "') + pynini.closure(pynini.accep(MINUS), 0, 1)
            + mixed_pattern + pynutil.insert('"')
        ).optimize()

        self.fst = (self.add_tokens(final_graph) | mixed).optimize()
