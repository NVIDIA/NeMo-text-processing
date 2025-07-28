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
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ar.taggers.money import ar_cur, maj_singular, min_plural, min_singular


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. "خمسة ريال سعودي" -> money { integer_part: "5" currency: "ر.س" }
        e.g. "سبعة دولار وتسعة وتسعون سنت"  -> money { integer_part: "7" currency: "$" fractional_part: "99" }

    Args:
        itn_cardinal_tagger: ITN Cardinal Tagger
    """

    def __init__(self, itn_cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal_graph = itn_cardinal_tagger.graph

        graph_unit = pynini.invert(maj_singular)
        graph_unit = pynutil.insert("currency: \"") + convert_space(graph_unit) + pynutil.insert("\"")

        graph_ar_cur = pynini.invert(ar_cur)
        graph_ar_cur = pynutil.insert("currency: \"") + convert_space(graph_ar_cur) + pynutil.insert("\"")

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        min_unit = pynini.project(min_singular | min_plural, "output")

        cents_standalone = (
            pynutil.insert("fractional_part: \"")
            + cardinal_graph @ add_leading_zero_to_double_digit
            + delete_space
            + pynutil.delete(min_unit)
            + pynutil.insert("\"")
        )

        optional_cents_standalone = pynini.closure(
            delete_space
            + pynutil.delete("و")
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + insert_space
            + cents_standalone,
            0,
            1,
        )
        graph_integer = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_unit
            + optional_cents_standalone
        )
        graph_integer_with_ar_cur = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_ar_cur
        )
        graph_decimal = pynutil.insert("currency: \"$\" integer_part: \"0\" ") + cents_standalone
        final_graph = graph_integer | graph_integer_with_ar_cur | graph_decimal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
