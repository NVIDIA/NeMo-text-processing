# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023, 2026, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_zero_or_one_space
from nemo_text_processing.text_normalization.se.utils import get_abs_path


class MoneyFst(GraphFst):
    """Classifies documented Northern Sámi integer currency expressions."""

    def __init__(self, cardinal: GraphFst, decimal=None, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency_nominative = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        currency_genitive = pynini.string_file(get_abs_path("data/money/currency_major_gen.tsv"))
        one = pynini.accep("1") @ cardinal.graph
        non_one = pynini.difference(pynini.project(cardinal.graph, "input"), "1") @ cardinal.graph
        separator = delete_zero_or_one_space
        optional_zero_fraction = pynini.closure(
            pynutil.delete(",") + (pynutil.delete("00") | pynutil.delete("–")), 0, 1
        )

        def integer_token(graph):
            return pynutil.insert('integer_part: "') + graph + pynutil.insert('" ')

        def currency_token(graph):
            return pynutil.insert('currency_maj: "') + graph + pynutil.insert('"')

        singular = integer_token(one) + optional_zero_fraction + separator + currency_token(currency_nominative)
        governed = integer_token(non_one) + optional_zero_fraction + separator + currency_token(currency_genitive)
        if not deterministic:
            governed |= (
                integer_token(non_one) + optional_zero_fraction + separator + currency_token(currency_nominative)
            )

        self.fst = self.add_tokens(singular | governed).optimize()
        self.graph = (
            one + optional_zero_fraction + separator + pynutil.insert(" ") + currency_nominative
            | non_one + optional_zero_fraction + separator + pynutil.insert(" ") + currency_genitive
        ).optimize()
