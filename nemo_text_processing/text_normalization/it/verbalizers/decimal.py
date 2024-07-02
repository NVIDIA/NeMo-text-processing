# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class DecimalFst(GraphFst):
    """
        Finite state transducer for classifying decimal, e.g.
                decimal { negative: "true" integer_part: "venti"  fractional_part: "trentaquattro" quantity: "miliardi" } ->
            meno venti virgola trentaquattro
                decimal { integer_part: "un milione" fractional_part: "zero zero zero" quantity: "milioni" preserve_order: true } -->
            un milione virgola zero zero zero
        decimal { integer_part: "due" quantity: "milioni" preserve_order: true } -->
            due milioni

    Args:
                deterministic: if True will provide a single transduction option,
                        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "meno ") + delete_space, 0, 1)
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        fractional_default = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        conjunction = pynutil.insert(" virgola ")
        fractional = conjunction + fractional_default

        quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(quantity, 0, 1)

        graph = optional_sign + pynini.union(
            (integer + quantity), (integer + delete_space + fractional + optional_quantity)
        )

        self.numbers_only_quantity = (
            optional_sign
            + pynini.union((integer + quantity), (integer + delete_space + fractional + quantity)).optimize()
        )

        self.graph = (graph + delete_preserve_order).optimize()

        graph += delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
