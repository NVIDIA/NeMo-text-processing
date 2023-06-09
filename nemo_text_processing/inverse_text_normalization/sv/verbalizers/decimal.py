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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
)
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "biljon" } -> -12,5006 biljon
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)
        plural_quantities_to_singular = pynini.string_map(
            [(f"{pfx}{ending}er", f"{pfx}{ending}") for pfx in ["m", "b", "tr"] for ending in ["iljon", "iljard"]]
        )
        delete_space = pynutil.delete(" ")
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)
        integer = (
            pynutil.delete("integer_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_integer = pynini.closure(integer + delete_space, 0, 1)
        fractional = (
            pynutil.delete("fractional_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_fractional = pynini.closure(pynutil.insert(",") + fractional, 0, 1)
        quantity = (
            pynutil.delete("quantity: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(NEMO_SPACE + quantity, 0, 1)
        graph = (optional_integer + optional_fractional + optional_quantity).optimize()
        optional_sign = pynini.closure("-", 0, 1)
        fix_singular = optional_sign + pynini.accep("1") + NEMO_SPACE + plural_quantities_to_singular
        self.numbers = optional_sign + graph
        self.numbers = self.numbers @ pynini.cdrewrite(fix_singular, "[BOS]", "[EOS]", NEMO_SIGMA)
        graph = self.numbers + delete_preserve_order
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
