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
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, load_labels


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure. Allows for plural forms of SI units.
        e.g. zwei kilogramm -> measure { cardinal { integer: "2" } units: "kg" }
        e.g. minus siebenundsechzig kilogramm -> measure { cardinal { negative: "-" integer: "67" } units: "kg" }
        e.g. eins komma fünf kilometer pro stunde -> measure { decimal { integer_part: "1" fractional_part: "5" } units: "km/h" }
        e.g. drei viertel ohm -> measure { fraction { 3/4 } units: "ω" }
        e.g. ein meter fünfzig -> measure { cardinal { integer: "1" } units: "m" morphosyntactic_features: "50" }
        e.g. sechzehner -> measure { cardinal { integer: "16" } morphosyntactic_features: "er" }
        e.g. vierzehnfach -> measure { cardinal { integer: "14" } morphosyntactic_features: "fach" }


    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst,
    ):
        super().__init__(name="measure", kind="classify")
        graph_SI_abbr = pynini.string_file(get_abs_path("data/measure/measurements.tsv"))
        plural_to_singular_exceptions = pynini.string_file(get_abs_path("data/measure/plural_to_singular.tsv"))
        plural_suffixes = load_labels(get_abs_path("data/measure/plural_suffixes.tsv"))
        denominator_markers = pynini.string_file(get_abs_path("data/measure/denominator_markers.tsv"))
        numeral_noun_suffixes = pynini.string_file(get_abs_path("data/measure/numeral_noun_suffixes.tsv"))
        adverbial_suffixes = pynini.string_file(get_abs_path("data/measure/adverbial_suffixes.tsv"))

        cardinals = pynutil.insert('integer: "') + cardinal.graph_no_exception + pynutil.insert('"')
        # leaves the first dozen verbalized
        cardinals_canonical = pynutil.insert('integer: "') + cardinal.graph + pynutil.insert('"')
        cardinals_no_separators = cardinal.graph_no_exception @ pynini.cdrewrite(
            pynutil.delete("."), "", "", NEMO_SIGMA
        )
        decimals = decimal.final_graph_wo_negative
        fractions = fraction.graph_wo_negative

        # Crudely depluralizes nominal forms.
        # Assumes the nominative case as the default, but it also applies to cases of morphological syncretism.
        graph_suffixes = pynini.union(
            *[
                NEMO_SIGMA + pynini.accep(lemma_ending) + pynutil.delete(plural_ending)
                for lemma_ending, plural_ending in plural_suffixes
            ]
        )
        depluralized = plurals._priority_union(plural_to_singular_exceptions, graph_suffixes, NEMO_SIGMA).optimize()
        graph_SI = (depluralized @ graph_SI_abbr) | graph_SI_abbr

        graph_denominator = denominator_markers + pynutil.delete(NEMO_SPACE) + graph_SI
        graph_unit = (
            pynutil.insert('units: "')
            + (
                graph_SI
                | graph_denominator
                | pynutil.add_weight(graph_SI + pynutil.delete(NEMO_SPACE) + graph_denominator, 0.01)
            )
            + pynutil.insert('"')
        )

        graph_cardinal = (
            pynutil.insert("cardinal {")
            + pynutil.insert(NEMO_SPACE)
            + cardinal.optional_minus_graph
            + cardinals
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
        )

        graph_cardinal_canonical = (
            pynutil.insert("cardinal {")
            + pynutil.insert(NEMO_SPACE)
            + cardinals_canonical
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
        )

        graph_decimal = (
            pynutil.insert("decimal {")
            + pynutil.insert(NEMO_SPACE)
            + cardinal.optional_minus_graph
            + decimals
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
        )

        # Graphs scrambled decimal and SI combination where the SI unit is placed before the fractional part
        scrambled_tail = cardinals_no_separators | pynutil.add_weight(decimal.graph_string_of_digits, 0.01)
        graph_decimal_scrambled = (
            cardinal.optional_minus_graph
            + pynutil.insert('integer_part: "')
            + cardinal.graph_no_exception
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + graph_unit
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert('fractional_part: "')
            + scrambled_tail
            + pynutil.insert('"')
        )

        graph_fraction = (
            pynutil.insert("fraction {")
            + pynutil.insert(NEMO_SPACE)
            + cardinal.optional_minus_graph
            + fractions
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
        )

        # Handles numeral nouns (e.g. fünfer -> fiver) as part of the MEASURE class
        graph_numeral_nouns = (
            graph_cardinal_canonical
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("morphosyntactic_features: ")
            + pynutil.insert('"')
            + numeral_noun_suffixes
            + pynutil.insert('"')
        )

        # Handles numeral adverbials created with affixed "-mal" or "-fach"
        graph_adverbials = (
            (graph_cardinal_canonical | graph_decimal)
            + (pynutil.delete("-") | pynutil.delete(NEMO_SPACE)).ques
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("morphosyntactic_features: ")
            + pynutil.insert('"')
            + adverbial_suffixes
            + pynutil.insert('"')
        )

        graphs_combined = graph_cardinal | pynutil.add_weight(graph_decimal, 0.01) | graph_fraction
        final_graph = (
            (graphs_combined + pynini.accep(NEMO_SPACE) + graph_unit)
            | graph_decimal_scrambled
            | graph_numeral_nouns
            | graph_adverbials
        )

        removes_empty_ints = pynutil.delete('cardinal { integer: "" } ')
        remove_empty_ints = pynini.cdrewrite(removes_empty_ints, "", "", NEMO_SIGMA)
        final_graph = final_graph @ remove_empty_ints

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
