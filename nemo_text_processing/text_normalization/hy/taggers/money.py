# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space
from nemo_text_processing.text_normalization.hy.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "15 $" -> money { "տասնհինգ դոլար" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.final_graph
        decimal_graph = decimal.fst

        unit = pynini.string_file(get_abs_path("data/currency.tsv"))

        weighted_delimiter = pynutil.add_weight(pynutil.delete(NEMO_SPACE), -100)
        optional_delimiter = pynini.closure(weighted_delimiter, 0, 1)
        graph_unit_singular = optional_delimiter + pynutil.insert(" currency: \"") + unit + pynutil.insert("\"")

        graph_decimal = decimal_graph + graph_unit_singular
        graph_cardinal = cardinal_graph + graph_unit_singular

        tagger_graph = graph_cardinal | graph_decimal

        integer = pynutil.delete("\"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer_cardinal = pynutil.delete("integer: ") + integer
        integer_part = pynutil.delete("integer_part: ") + integer

        unit = (
            pynutil.delete("currency: ")
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        unit = pynini.accep(NEMO_SPACE) + unit

        verbalizer_graph_cardinal = integer_cardinal + unit

        optional_fractional_part = pynini.closure(pynutil.delete("fractional_part: ") + integer, 0, 1)
        optional_quantity = pynini.closure(pynini.accep(NEMO_SPACE) + pynutil.delete("quantity: ") + integer, 0, 1)

        verbalizer_graph_decimal = (
            pynutil.delete('decimal { ')
            + integer_part
            + delete_space
            + pynutil.insert(" ամբողջ ")
            + optional_fractional_part
            + delete_space
            + optional_quantity
            + delete_space
            + pynutil.delete(" }")
            + unit
        )

        verbalizer_graph = verbalizer_graph_cardinal | verbalizer_graph_decimal

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(
            pynutil.insert("integer_part: \"") + self.final_graph + pynutil.insert("\"")
        ).optimize()
