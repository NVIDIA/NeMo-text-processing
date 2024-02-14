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

from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "تسعة" currency_maj: "يورو" preserve_order: true} -> "تسعة يورو"
        money { integer_part: "تسعة" currency_maj: "دولار" preserve_order: true} -> "تسعة دولار"
        money { integer_part: "خمسة" currency_maj: "دينار كويتي"} -> "خمسة دينار كويتي"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        keep_space = pynini.accep(" ")

        maj = pynutil.delete("currency_maj: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        min = pynutil.delete("currency_min: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        fractional_part = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        integer_part = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        add_and = pynutil.insert(" و")

        #  *** currency_maj
        graph_integer = maj + keep_space + integer_part

        #  *** currency_maj + (***) (و) *** current_min
        graph_integer_with_minor = (
            integer_part
            + keep_space
            + maj
            + delete_space
            + add_and
            + fractional_part
            + delete_space
            + pynini.closure(keep_space + min, 0, 1)
            + delete_preserve_order
        )
        # this graph fix word order from dollar three (دولار تسعة)--> three dollar (تسعة دولار)
        graph_integer_no_minor = integer_part + keep_space + maj + delete_space + delete_preserve_order
        # *** current_min
        graph_minor = fractional_part + keep_space + delete_space + min + delete_preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_minor | graph_integer_no_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
