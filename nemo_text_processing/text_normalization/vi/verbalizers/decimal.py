# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_COMMA_VI,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese decimal numbers, e.g.
        decimal { negative: "true" integer_part: "mười hai" fractional_part: "năm" quantity: "tỷ" } -> âm mười hai phẩy năm tỷ
        decimal { integer_part: "tám trăm mười tám" fractional_part: "ba không ba" } -> tám trăm mười tám phẩy ba không ba
        decimal { integer_part: "không" fractional_part: "hai" quantity: "triệu" } -> không phẩy hai triệu

    Args:
        cardinal: CardinalFst instance for handling integer verbalization
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        # Basic components
        integer = pynutil.delete("integer_part:") + cardinal.integer
        fractional = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Negative handling
        negative = pynini.cross("negative: \"true\"", "âm ")
        if not deterministic:
            negative |= pynini.cross("negative: \"true\"", "trừ ")
        optional_negative = pynini.closure(negative + delete_space, 0, 1)

        # Simple patterns
        simple_integer = integer

        integer_with_quantity = integer + delete_space + insert_space + quantity

        decimal_with_comma = (
            integer + delete_space + insert_space + pynutil.insert(NEMO_COMMA_VI) + insert_space + fractional
        )

        decimal_with_quantity = (
            integer
            + delete_space
            + insert_space
            + pynutil.insert(NEMO_COMMA_VI)
            + insert_space
            + fractional
            + delete_space
            + insert_space
            + quantity
        )

        fractional_only = (
            pynini.closure(integer + delete_space + insert_space, 0, 1)
            + pynutil.insert(NEMO_COMMA_VI)
            + insert_space
            + fractional
        )

        # Group all patterns
        all_patterns = pynini.union(
            simple_integer, integer_with_quantity, decimal_with_comma, decimal_with_quantity, fractional_only
        )

        # Combine with negative handling
        graph = optional_negative + all_patterns

        self.numbers = graph
        self.fst = self.delete_tokens(graph).optimize()
