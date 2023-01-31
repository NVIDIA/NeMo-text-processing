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
from nemo_text_processing.text_normalization.de.taggers.money import maj_singular, min_plural, min_singular
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. elf euro und vier cent -> money { integer_part: "11" fractional_part: 04 currency: "€" }

    Args:
        itn_cardinal_tagger: ITN Cardinal Tagger
        itn_decimal_tagger: ITN Decimal Tagger
    """

    def __init__(self, itn_cardinal_tagger: GraphFst, itn_decimal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = (
            pynini.cdrewrite(pynini.cross(pynini.union("ein", "eine"), "eins"), "[BOS]", "[EOS]", NEMO_SIGMA)
            @ itn_cardinal_tagger.graph_no_exception
        )
        graph_decimal_final = itn_decimal_tagger.final_graph_wo_negative

        graph_unit = pynini.invert(maj_singular)
        graph_unit = pynutil.insert("currency: \"") + convert_space(graph_unit) + pynutil.insert("\"")

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        min_unit = pynini.project(min_singular | min_plural, "output")
        # elf euro (und) vier cent, vier cent
        cents_standalone = (
            pynutil.insert("fractional_part: \"")
            + cardinal_graph @ add_leading_zero_to_double_digit
            + delete_space
            + pynutil.delete(min_unit)
            + pynutil.insert("\"")
        )

        optional_cents_standalone = pynini.closure(
            delete_space
            + pynini.closure(pynutil.delete("und") + delete_space, 0, 1)
            + insert_space
            + cents_standalone,
            0,
            1,
        )
        # elf euro vierzig, only after integer
        optional_cents_suffix = pynini.closure(
            delete_extra_space
            + pynutil.insert("fractional_part: \"")
            + pynutil.add_weight(cardinal_graph @ add_leading_zero_to_double_digit, -0.7)
            + pynutil.insert("\""),
            0,
            1,
        )

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_unit
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_decimal = graph_decimal_final + delete_extra_space + graph_unit
        graph_decimal |= pynutil.insert("currency: \"€\" integer_part: \"0\" ") + cents_standalone
        final_graph = graph_integer | graph_decimal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
