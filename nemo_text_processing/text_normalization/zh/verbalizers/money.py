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


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction e.g.
        tokens { money { integer: "二" currency: "$"} } -> 二美元
        tokens { money { integer: "三" major_unit: "块"} } -> 三块
        tokens { money { currency: "$" integer: "二" } } -> 二美元
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # components to combine to make graphs
        number_component = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        currency_component = pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        decimal_component = decimal.decimal_component
        unit_only_component = (
            (pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\""))
            | (pynutil.delete("currency_maj: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\""))
            | (pynutil.delete("currency_min: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\""))
        )

        # graphs
        graph_regular_money = number_component + delete_space + currency_component
        graph_unit_money = pynini.closure(
            (number_component + delete_space + unit_only_component + pynini.closure(delete_space))
        )
        graph_decimal_money = decimal_component + delete_space + currency_component

        graph_suffix = (
            number_component
            + delete_space
            + pynutil.delete("quantity: \"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
            + currency_component
        )

        graph = graph_unit_money | graph_regular_money | graph_decimal_money | graph_suffix

        final_graph = graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
