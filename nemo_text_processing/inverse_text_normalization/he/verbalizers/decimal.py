# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_ALPHA_HE, GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_NOT_QUOTE, delete_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal,
    e.g. decimal { integer_part: "0"  fractional_part: "33" } -> 0.33
    e.g. decimal { negative: "true" integer_part: "400"  fractional_part: "323" } -> -400.323
    e.g. decimal { integer_part: "4"  fractional_part: "5" quantity: "מיליון" } -> 4.5 מיליון

    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")
        optionl_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)

        # Need parser to group digits by threes
        exactly_three_digits = NEMO_DIGIT**3
        at_most_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        # Thousands separator
        group_by_threes = at_most_three_digits + (pynutil.insert(",") + exactly_three_digits).closure()

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        integer = integer @ group_by_threes

        optional_integer = pynini.closure(integer + delete_space, 0, 1)

        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)

        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)

        # Keep the prefix if exists and add a dash
        optional_prefix = pynini.closure(
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_ALPHA_HE, 1)
            + pynutil.insert("-")
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )

        graph = optional_prefix + optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        graph = optionl_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
