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
    PERIOD,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals. The tagger only stores the digits after
    the point; the verbalizer inserts the word for the decimal point (புள்ளி).
    Both Latin (0-9) and Tamil (௦-௯) digits are accepted.
        e.g. "1.5" -> decimal { integer_part: "ஒன்று" fractional_part: "ஐந்து" }
        e.g. "-2.67" -> decimal { negative: "true" integer_part: "இரண்டு" fractional_part: "ஆறு ஏழு" }
        e.g. "00.5" -> decimal { integer_part: "சுழியம் சுழியம்" fractional_part: "ஐந்து" }
        e.g. ".5" -> decimal { fractional_part: "ஐந்து" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        delete_point = pynutil.delete(PERIOD)
        zeros = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        digits = pynini.closure(NEMO_ALL_DIGIT, 1)

        optional_sign = pynini.closure(pynutil.insert("negative: ") + pynini.cross(MINUS, '"true" '), 0, 1)

        zero = pynini.compose(pynini.project(zeros, "input"), NEMO_ALL_DIGIT).optimize()
        nonzero = pynini.difference(NEMO_ALL_DIGIT, zero).optimize()
        valid_int = (zero | nonzero + pynini.closure(NEMO_ALL_DIGIT | pynini.accep(COMMA))).optimize()

        integer = pynini.compose(valid_int, cardinal.final_graph).optimize()
        fraction = pynini.compose(digits, cardinal.single_digits_graph).optimize()
        leading_zero = pynini.compose(zero + digits, cardinal.single_digits_graph).optimize()

        integer_part = pynutil.insert('integer_part: "') + integer + pynutil.insert('"')
        leading_zero_part = pynutil.insert('integer_part: "') + leading_zero + pynutil.insert('"')
        fractional_part = pynutil.insert('fractional_part: "') + fraction + pynutil.insert('"')

        with_int = integer_part + delete_point + insert_space + fractional_part
        without_int = delete_point + fractional_part
        leading_zero_decimal = leading_zero_part + delete_point + insert_space + fractional_part

        final_graph = optional_sign + (leading_zero_decimal | with_int | without_int)

        self.fst = self.add_tokens(final_graph).optimize()
