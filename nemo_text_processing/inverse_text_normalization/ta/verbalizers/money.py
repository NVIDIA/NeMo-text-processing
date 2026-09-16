# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    delete_preserve_order,
    delete_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "50" currency: "₹" fractional_part: "50" } -> ₹50.50
        money { integer_part: "50" currency: "₹" morphosyntactic_features: "க்கு" } -> ₹50க்கு
        money { negative: "true" integer_part: "500" currency: "₹" } -> -₹500
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)
        currency = pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        # A single spoken minor digit is tens of paise: ஐந்து பைசா is ₹0.05, not ₹0.5.
        two_digits = pynini.union(NEMO_DIGIT + NEMO_DIGIT, pynutil.insert("0") + NEMO_DIGIT)
        fraction = pynutil.delete("fractional_part: \"") + two_digits + pynutil.delete("\"")
        suffix = (
            pynutil.delete("morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )
        self.graph = (
            optional_sign
            + currency
            + delete_space
            + integer
            + pynini.closure(delete_space + pynutil.insert(".") + fraction, 0, 1)
            + pynini.closure(delete_space + suffix, 0, 1)
            + delete_preserve_order
        )
        self.fst = self.delete_tokens(self.graph).optimize()
