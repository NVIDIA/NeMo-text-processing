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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_preserve_order


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { currency_maj: "euró" integer_part: "egy"} -> "egy euró"
        money { currency_maj: "euró" integer_part: "egy" fractional_part: "egy ezred"} -> "egy egész egy ezred euro"
        money { integer_part: "egy" currency_maj: "font" fractional_part: "negyven" preserve_order: true} -> "egy font negyven"
        money { integer_part: "egy" currency_maj: "font" fractional_part: "negyven" currency_min: "penny" preserve_order: true} -> "egy font negyven penny"
        money { fractional_part: "egy" currency_min: "penny" preserve_order: true} -> "egy penny"
        money { currency_maj: "font" integer_part: "nulla" fractional_part: "null eins" quantity: "millió"} -> "null egész egy század millió font"

    Args:
        decimal: GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        keep_space = pynini.accep(" ")

        maj = pynutil.delete("currency_maj: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        min = pynutil.delete("currency_min: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        fractional_part = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        integer_part = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_add_and = pynini.closure(pynutil.insert("és "), 0, 1)

        #  *** currency_maj
        graph_integer = integer_part + keep_space + maj

        #  *** currency_maj + (***) | ((und) *** current_min)
        minor_part = fractional_part | (fractional_part + keep_space + min)
        if not deterministic:
            minor_part |= optional_add_and + fractional_part + keep_space + min
        graph_integer_with_minor = integer_part + keep_space + maj + keep_space + minor_part + delete_preserve_order

        # *** komma *** currency_maj
        graph_decimal = decimal.numbers + keep_space + maj
        graph_decimal |= decimal.numbers + keep_space + maj + delete_preserve_order

        # *** current_min
        graph_minor = fractional_part + keep_space + min + delete_preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
