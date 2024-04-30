# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { integer_part: "零" fractional_part: "五" } -> 零点五
        decimal { integer_part: "零" fractional_part: "五" quantity: "万" } -> 零点五万
        decimal { positive: "正" integer_part: "零" fractional_part: "五" quantity: "万" } -> 正零点五万
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        fractional = pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("负")
            + pynutil.delete("\"")
        )

        graph = integer + delete_space + pynutil.insert("点") + fractional
        self.decimal_regular = graph
        graph_quantity = graph + delete_space + quantity
        graph_regular = graph | graph_quantity

        graph_sign = sign + delete_space + graph_regular

        final_graph = graph_regular | graph_sign
        self.decimal_component = final_graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
