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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing Japanese money.

    Example:
        money { integer_part: "百" currency_maj: "円" preserve_order: true } -> 百円
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        field_value = pynini.closure(NEMO_NOT_QUOTE, 1)

        integer_part = pynutil.delete('integer_part: "') + field_value + pynutil.delete('"')

        quantity = pynini.closure(
            delete_space + pynutil.delete('quantity: "') + field_value + pynutil.delete('"'),
            0,
            1,
        )

        currency_major = delete_space + pynutil.delete('currency_maj: "') + field_value + pynutil.delete('"')

        fractional_part = pynini.closure(
            delete_space
            + pynutil.delete('fractional_part: "')
            + field_value
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete('currency_min: "')
            + field_value
            + pynutil.delete('"'),
            0,
            1,
        )

        graph = integer_part + quantity + currency_major + fractional_part + delete_preserve_order

        self.fst = self.delete_tokens(graph.optimize()).optimize()
