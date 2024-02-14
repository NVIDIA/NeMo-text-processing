# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import NEMO_CHAR, GraphFst, delete_space


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "12" fractional_part: "05" currency: "$" } -> $12.05

    Args:
        decimal: ITN Decimal verbalizer
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        # optionl_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)
        # integer = (
        #     pynutil.delete("integer_part:")
        #     + delete_space
        #     + pynutil.delete("\"")
        #     + pynini.closure(NEMO_NOT_QUOTE, 1)
        #     + pynutil.delete("\"")
        # )
        # optional_integer = pynini.closure(integer + delete_space, 0, 1)
        # fractional = (
        #     pynutil.insert(".")
        #     + pynutil.delete("fractional_part:")
        #     + delete_space
        #     + pynutil.delete("\"")
        #     + pynini.closure(NEMO_NOT_QUOTE, 1)
        #     + pynutil.delete("\"")
        # )
        # optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        # quantity = (
        #     pynutil.delete("quantity:")
        #     + delete_space
        #     + pynutil.delete("\"")
        #     + pynini.closure(NEMO_NOT_QUOTE, 1)
        #     + pynutil.delete("\"")
        # )
        # optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        # graph = optional_integer + optional_fractional + optional_quantity
        # graph = optionl_sign + graph
        # graph= unit + delete_space + optional_integer + optional_fractional
        # #graph= optional_integer + optional_fractional + unit
        graph = unit + delete_space + decimal.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
        self.unit = unit
