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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_zero_or_one_space
from nemo_text_processing.text_normalization.se.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """Classifies documented Northern Sámi integer currency expressions."""

    def __init__(self, cardinal: GraphFst, decimal=None, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency_nominative = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        currency_genitive = pynini.string_file(get_abs_path("data/money/currency_major_gen.tsv"))
        minor_standalone = pynini.string_file(get_abs_path("data/money/currency_minor_standalone.tsv"))
        minor_standalone_genitive = pynini.string_file(get_abs_path("data/money/currency_minor_standalone_gen.tsv"))
        major_forms = {
            "nom": dict(load_labels(get_abs_path("data/money/currency_major.tsv"))),
            "gen": dict(load_labels(get_abs_path("data/money/currency_major_gen.tsv"))),
        }
        minor_forms = {
            "nom": dict(load_labels(get_abs_path("data/money/currency_minor.tsv"))),
            "gen": dict(load_labels(get_abs_path("data/money/currency_minor_gen.tsv"))),
        }
        cardinal_graph = cardinal.graphs["nom_sg"]
        one = pynini.accep("1") @ cardinal_graph
        non_one = pynini.difference(pynini.project(cardinal_graph, "input"), "1") @ cardinal_graph
        separator = delete_zero_or_one_space
        optional_zero_fraction = pynini.closure(
            pynutil.delete(",") + (pynutil.delete("00") | pynutil.delete("-") | pynutil.delete("–")), 0, 1
        )

        def integer_token(graph):
            return pynutil.insert('integer_part: "') + graph + pynutil.insert('" ')

        def currency_token(graph):
            return pynutil.insert('currency_maj: "') + graph + pynutil.insert('"')

        def fractional_token(graph):
            return pynutil.insert('fractional_part: "') + graph + pynutil.insert('" ')

        def minor_token(graph):
            return pynutil.insert('currency_min: "') + graph + pynutil.insert('" preserve_order: true')

        fractional_one = pynutil.delete(",0") + ("1" @ cardinal_graph)
        fractional_non_one = pynutil.delete(",") + (
            pynutil.delete("0") + ((NEMO_DIGIT - "0" - "1") @ cardinal_graph)
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ cardinal_graph
        )

        def number_phrase(graph, conjunction):
            return (pynutil.insert("ja ") + graph) if conjunction else graph

        def fractional_fields(major_form, minor_form, number_graph, tokenized, conjunction):
            phrase = number_phrase(number_graph, conjunction)
            if tokenized:
                return (
                    currency_token(pynutil.insert(major_form))
                    + pynutil.insert(" ")
                    + fractional_token(phrase)
                    + minor_token(pynutil.insert(minor_form))
                )
            return (
                pynutil.insert(" ")
                + pynutil.insert(major_form)
                + pynutil.insert(" ")
                + phrase
                + pynutil.insert(" ")
                + pynutil.insert(minor_form)
            )

        def suffix_fractional(integer_graph, major_case, tokenized=True, conjunction=True):
            alternatives = []
            for symbol, major_form in major_forms[major_case].items():
                if symbol in minor_forms["nom"]:
                    fields = fractional_fields(
                        major_form, minor_forms["nom"][symbol], fractional_one, tokenized, conjunction
                    )
                    alternatives.append(integer_graph + fields + separator + pynutil.delete(symbol))
                if symbol in minor_forms["gen"]:
                    fields = fractional_fields(
                        major_form, minor_forms["gen"][symbol], fractional_non_one, tokenized, conjunction
                    )
                    alternatives.append(integer_graph + fields + separator + pynutil.delete(symbol))
            return pynini.union(*alternatives)

        def prefix_integer(integer_graph, major_case, tokenized=True):
            alternatives = []
            for symbol, major_form in major_forms[major_case].items():
                prefix = pynutil.delete(symbol) + separator
                if tokenized:
                    output = integer_token(integer_graph) + currency_token(pynutil.insert(major_form))
                else:
                    output = integer_graph + pynutil.insert(" ") + pynutil.insert(major_form)
                alternatives.append(prefix + output)
            return pynini.union(*alternatives)

        def prefix_fractional(integer_graph, major_case, tokenized=True, conjunction=True):
            alternatives = []
            for symbol, major_form in major_forms[major_case].items():
                integer = integer_token(integer_graph) if tokenized else integer_graph
                prefix = pynutil.delete(symbol) + separator + integer
                if symbol in minor_forms["nom"]:
                    fields = fractional_fields(
                        major_form, minor_forms["nom"][symbol], fractional_one, tokenized, conjunction
                    )
                    alternatives.append(prefix + fields)
                if symbol in minor_forms["gen"]:
                    fields = fractional_fields(
                        major_form, minor_forms["gen"][symbol], fractional_non_one, tokenized, conjunction
                    )
                    alternatives.append(prefix + fields)
            return pynini.union(*alternatives)

        singular = integer_token(one) + optional_zero_fraction + separator + currency_token(currency_nominative)
        governed = integer_token(non_one) + optional_zero_fraction + separator + currency_token(currency_genitive)
        singular |= prefix_integer(one, "nom") + optional_zero_fraction
        governed |= prefix_integer(non_one, "gen") + optional_zero_fraction
        fractional = suffix_fractional(integer_token(one), "nom")
        fractional |= suffix_fractional(integer_token(non_one), "gen")
        fractional |= prefix_fractional(one, "nom")
        fractional |= prefix_fractional(non_one, "gen")
        if not deterministic:
            fractional |= suffix_fractional(integer_token(one), "nom", conjunction=False)
            fractional |= suffix_fractional(integer_token(non_one), "gen", conjunction=False)
            fractional |= prefix_fractional(one, "nom", conjunction=False)
            fractional |= prefix_fractional(non_one, "gen", conjunction=False)
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
            | suffix_fractional(one, "nom", tokenized=False)
            | suffix_fractional(non_one, "gen", tokenized=False)
            | prefix_fractional(one, "nom", tokenized=False)
            | prefix_fractional(non_one, "gen", tokenized=False)
            | prefix_integer(one, "nom", tokenized=False) + optional_zero_fraction
            | prefix_integer(non_one, "gen", tokenized=False) + optional_zero_fraction
            | one + separator + pynutil.insert(" ") + minor_standalone
            | non_one + separator + pynutil.insert(" ") + minor_standalone_genitive
        ).optimize()
        if not deterministic:
            self.graph |= suffix_fractional(one, "nom", tokenized=False, conjunction=False)
            self.graph |= suffix_fractional(non_one, "gen", tokenized=False, conjunction=False)
            self.graph |= prefix_fractional(one, "nom", tokenized=False, conjunction=False)
            self.graph |= prefix_fractional(non_one, "gen", tokenized=False, conjunction=False)
            self.graph = self.graph.optimize()
