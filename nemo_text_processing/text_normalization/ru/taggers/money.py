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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "5руб." -> money { "пять рублей" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.cardinal_numbers_default
        decimal_graph = decimal.final_graph

        unit_singular = pynini.string_file(get_abs_path("data/currency/currency_singular.tsv"))
        unit_plural = pynini.string_file(get_abs_path("data/currency/currency_plural.tsv"))

        # adding weight to make sure the space is preserved for ITN
        optional_delimiter = pynini.closure(pynutil.add_weight(pynini.cross(NEMO_SPACE, ""), -100), 0, 1)
        graph_unit_singular = (
            optional_delimiter + pynutil.insert(" currency: \"") + unit_singular + pynutil.insert("\"")
        )
        graph_unit_plural = optional_delimiter + pynutil.insert(" currency: \"") + unit_plural + pynutil.insert("\"")

        one = pynini.compose(pynini.accep("1"), cardinal_graph).optimize()
        singular_graph = pynutil.insert("integer_part: \"") + one + pynutil.insert("\"") + graph_unit_singular

        graph_decimal = decimal_graph + graph_unit_plural

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + pynutil.insert("\"")
            + (graph_unit_plural)
        )

        graph_integer |= singular_graph
        tagger_graph = (graph_integer.optimize() | graph_decimal.optimize()).optimize()

        # verbalizer
        integer = pynutil.delete("\"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer_part = pynutil.delete("integer_part: ") + integer

        unit = (
            pynutil.delete("currency: ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        unit = pynini.accep(NEMO_SPACE) + unit

        verbalizer_graph_cardinal = (integer_part + unit).optimize()

        fractional_part = pynutil.delete("fractional_part: ") + integer
        optional_quantity = pynini.closure(pynini.accep(NEMO_SPACE) + pynutil.delete("quantity: ") + integer, 0, 1)

        verbalizer_graph_decimal = (
            pynutil.delete('decimal { ')
            + integer_part
            + pynini.accep(" ")
            + fractional_part
            + optional_quantity
            + pynutil.delete(" }")
            + unit
        )

        verbalizer_graph = (verbalizer_graph_cardinal | verbalizer_graph_decimal).optimize()

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(
            pynutil.insert("integer_part: \"") + self.final_graph + pynutil.insert("\"")
        ).optimize()
