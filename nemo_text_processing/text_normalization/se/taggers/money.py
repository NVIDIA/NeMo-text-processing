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
from nemo_text_processing.text_normalization.se.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """Classifies documented Northern Sámi integer currency expressions."""

    def __init__(self, cardinal: GraphFst, decimal=None, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency_nominative = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        currency_genitive = pynini.string_file(get_abs_path("data/money/currency_major_gen.tsv"))
        minor_standalone = pynini.string_file(get_abs_path("data/money/currency_minor_standalone.tsv"))
        minor_standalone_genitive = pynini.string_file(
            get_abs_path("data/money/currency_minor_standalone_gen.tsv")
        )
        major_forms = {
            "nom": dict(load_labels(get_abs_path("data/money/currency_major.tsv"))),
            "gen": dict(load_labels(get_abs_path("data/money/currency_major_gen.tsv"))),
        }
        minor_forms = {
            "nom": dict(load_labels(get_abs_path("data/money/currency_minor.tsv"))),
            "gen": dict(load_labels(get_abs_path("data/money/currency_minor_gen.tsv"))),
        }
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

        def fractional_token(graph):
            return pynutil.insert('fractional_part: "') + graph + pynutil.insert('" ')

        def minor_token(graph):
            return pynutil.insert('currency_min: "') + graph + pynutil.insert('" preserve_order: true')

        def fractional_currency(major_case, tokenized, conjunction=True):
            pairs = []
            for number in range(1, 100):
                minor_case = "nom" if number == 1 else "gen"
                number_input = str(number) if number >= 10 else f"0{number}"
                number_output = pynini.shortestpath(pynini.accep(str(number)) @ cardinal.graph).string()
                for symbol in major_forms[major_case].keys() & minor_forms[minor_case].keys():
                    major_form = major_forms[major_case][symbol]
                    minor_form = minor_forms[minor_case][symbol]
                    number_phrase = f"ja {number_output}" if conjunction else number_output
                    if tokenized:
                        output = (
                            f'currency_maj: "{major_form}" fractional_part: "{number_phrase}" '
                            f'currency_min: "{minor_form}" preserve_order: true'
                        )
                    else:
                        output = f" {major_form} {number_phrase} {minor_form}"
                    pairs.append((f",{number_input}{symbol}", output))
                    pairs.append((f",{number_input} {symbol}", output))
            return pynini.string_map(pairs)

        singular = integer_token(one) + optional_zero_fraction + separator + currency_token(currency_nominative)
        governed = integer_token(non_one) + optional_zero_fraction + separator + currency_token(currency_genitive)
        fractional = integer_token(one) + fractional_currency("nom", tokenized=True)
        fractional |= integer_token(non_one) + fractional_currency("gen", tokenized=True)
        if not deterministic:
            fractional |= integer_token(one) + fractional_currency("nom", tokenized=True, conjunction=False)
            fractional |= integer_token(non_one) + fractional_currency("gen", tokenized=True, conjunction=False)
        minor_singular = fractional_token(one) + separator + minor_token(minor_standalone)
        minor_governed = fractional_token(non_one) + separator + minor_token(minor_standalone_genitive)
        if not deterministic:
            governed |= (
                integer_token(non_one) + optional_zero_fraction + separator + currency_token(currency_nominative)
            )

        self.fst = self.add_tokens(singular | governed | fractional | minor_singular | minor_governed).optimize()
        self.graph = (
            one + optional_zero_fraction + separator + pynutil.insert(" ") + currency_nominative
            | non_one + optional_zero_fraction + separator + pynutil.insert(" ") + currency_genitive
            | one + fractional_currency("nom", tokenized=False)
            | non_one + fractional_currency("gen", tokenized=False)
            | one + separator + pynutil.insert(" ") + minor_standalone
            | non_one + separator + pynutil.insert(" ") + minor_standalone_genitive
        ).optimize()
        if not deterministic:
            self.graph |= one + fractional_currency("nom", tokenized=False, conjunction=False)
            self.graph |= non_one + fractional_currency("gen", tokenized=False, conjunction=False)
            self.graph = self.graph.optimize()
