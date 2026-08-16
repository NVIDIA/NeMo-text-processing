# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
)
from nemo_text_processing.text_normalization.sv.graph_utils import ensure_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "biljoner" } -> -12,5006 biljoner
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        fractional = pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        quantity = pynutil.delete("quantity: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        number = pynini.union(
            optional_sign + integer,
            optional_sign + integer + pynini.cross(" ", ",") + fractional,
            pynutil.insert(",") + fractional,
        )
        number_quantity = number + ensure_space + quantity
        optional_delete_preserve_order = pynini.closure(delete_preserve_order, 0, 1)

        graph = (number | number_quantity | quantity).optimize()
        self.graph = graph
        graph = self.graph + optional_delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
