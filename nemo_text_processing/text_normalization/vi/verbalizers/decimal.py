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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


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

        # Handle negative sign - Vietnamese uses "âm" for negative numbers
        self.optional_sign = pynini.cross("negative: \"true\"", "âm ")
        if not deterministic:
            # Alternative ways to say negative in Vietnamese
            self.optional_sign |= pynini.cross("negative: \"true\"", "trừ ")

        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)

        self.integer = pynutil.delete("integer_part:") + cardinal.integer
        self.optional_integer = pynini.closure(self.integer + delete_space + insert_space, 0, 1)

        # Handle fractional part - Vietnamese uses "phẩy" (comma) instead of "point"
        self.fractional_default = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        self.fractional = pynutil.insert("phẩy ") + self.fractional_default

        self.quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        self.optional_quantity = pynini.closure(self.quantity, 0, 1)

        graph = self.optional_sign + (
            self.integer
            | (self.integer + self.quantity)
            | (self.optional_integer + self.fractional + self.optional_quantity)
        )

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
