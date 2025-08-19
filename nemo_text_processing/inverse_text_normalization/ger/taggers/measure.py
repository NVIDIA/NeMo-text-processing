# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pynini.examples import plurals
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    NEMO_SPACE,
    NEMO_SIGMA,
)

graph_SI_abbr = pynini.string_file(get_abs_path("data/measure/units.tsv"))
singular_2_plural_hardcoded = pynini.string_file(
    get_abs_path("data/measure/units_to_singular.tsv")
)


# The function below crudely depluralizes nominal forms.
# It assumes the nominative case as the default, but it will also apply to cases of moprhological syncretism.
# To produce plural forms replace deletions with insertions.
def depluralize():
    # Removes "-n" from the lemmas ending in "-e"
    n_suffix = NEMO_SIGMA + pynini.accep("e") + pynutil.delete("n")
    # Removes "-en" from the lemmas ending in:
    # "ent", "and", "ant", "ist", "or", "ion", "ik", "heit", "keit", "schaft", "tät", "ung"
    en_suffix = (
        NEMO_SIGMA
        + pynini.union(
            "ent",
            "and",
            "ant",
            "ist",
            "or",
            "ion",
            "ik",
            "heit",
            "keit",
            "schaft",
            "tät",
            "ung",
        )
        + pynutil.delete("en")
    )
    # Removes "-en" or "-e" from the lemmas ending in "-in"
    nen_suffix = (
        NEMO_SIGMA + pynini.accep("in") + (pynutil.delete("e") | pynutil.delete("en"))
    )
    # Handles the pluralzation of borrowings based on non-germanic lemma suffixes
    borrowings = NEMO_SIGMA + pynini.union("ma", "um", "us") + pynutil.delete("en")
    # Appends "-e" to nouns ending in "eur", "ich", "ier", "ig", "ling", "ör"
    e_suffix = (
        NEMO_SIGMA
        + pynini.union("eur", "ich", "ier", "ig", "ling", "ör")
        + pynutil.delete("e")
    )
    # Appends "-s" to nouns ending in vowels except "-e" and umlauts ("a", "i", "o", "u", "y")
    s_suffix = NEMO_SIGMA + pynini.union("a", "i", "o", "u", "y") + pynutil.delete("s")
    graph_depluralized = plurals._priority_union(
        singular_2_plural_hardcoded,
        pynini.union(n_suffix, en_suffix, nen_suffix, borrowings, e_suffix, s_suffix),
        NEMO_SIGMA,
    ).optimize()
    return graph_depluralized


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure. Allows for plural forms of SI units.
        e.g. zwei kilogramm -> measure { cardinal: "2" units: "kg" }
        e.g. eins komma fünf kilometer pro stunde -> measure { decimal { integer_part: "1" fractional_part: "5" } units: "km/h" }
        e.g. drei viertel ohm -> measure { fraction: "3/4" units: "ω" }
        e.g. minus zwei millionen quadrat meter pro sekunde -> measure { cardinal: "-2.000.000" units: "m²/s" }`
        e.g. Sechzehner -> measure { cardinal: "16" morphosyntactic_features: "er" }
        e.g. Kapitel vier -> measure { units: "Kapitel" cardinal: "4" }
        e.g. vierzehnfach -> measure { cardinal: "14" morphosyntactic_features: "fach" }
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst,
    ):
        super().__init__(name="measure", kind="classify")

        # Handles "ein" and its declensions
        morphemes_for_one = (
            pynini.accep("ein") + pynini.accep("e").ques + pynini.union(*"rnsm").ques
        )
        declensions_of_one = pynini.cross(morphemes_for_one, "1")
        graph_declensions_of_one = (
            pynutil.insert('integer: "') + declensions_of_one + pynutil.insert('"')
        )

        cardinals = cardinal.all_cardinals_graph | graph_declensions_of_one
        cardinals_canonical = (
            cardinal.canonical_cardinals_graph
        )  # leaves the first dozen verbalized
        cardinals_no_separators = fraction.cardinals_no_separators
        decimals = decimal.graph_decimal
        fractions = fraction.graph_fraction
        graph_SI = (depluralize() @ graph_SI_abbr) | graph_SI_abbr

        # Handles units in the denominator
        graph_pro = pynini.cross("pro", "/") + pynutil.delete(NEMO_SPACE) + graph_SI
        graph_unit = (
            pynutil.insert('units: "')
            + (
                graph_SI
                | graph_pro
                | pynutil.add_weight(
                    graph_SI + pynutil.delete(NEMO_SPACE) + graph_pro, 0.01
                )
            )
            + pynutil.insert('"')
        )

        # Graphs cardinals
        graph_cardinal = (
            pynutil.insert("cardinal {")
            + pynutil.insert(NEMO_SPACE)
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

        # Graphs decimals
        # Curly braces are added to the decimal graph to permit later use of the decimal verbalizer
        graph_decimal = (
            pynutil.insert("decimal {")
            + pynutil.insert(NEMO_SPACE)
            + decimals
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
        )

        # Graphs scrambled decimal and SI combination where the SI unit is placed before the fractional part
        scrambled_decimal = (
            pynutil.insert("morphosyntactic_features: ")
            + pynutil.insert('"')
            + cardinals_no_separators
            + pynutil.insert('"')
        )
        graph_decimal_scrambled = (
            graph_cardinal
            + pynini.accep(NEMO_SPACE)
            + graph_unit
            + pynini.accep(NEMO_SPACE)
            + scrambled_decimal
        )

        # Graphs fractions
        graph_fraction = (
            pynutil.insert("fraction {")
            + pynutil.insert(NEMO_SPACE)
            + fractions
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
        )

        # Handles numeral nouns (e.g. fünfer -> fiver) as part of the MEASURE class
        suffix = pynini.accep("er") + pynini.union(*"nmrs").ques
        graph_numeral_nouns = (
            graph_cardinal_canonical
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("morphosyntactic_features: ")
            + pynutil.insert('"')
            + suffix
            + pynutil.insert('"')
        )

        # Handles numeral adverbials created with affixed "-mal" or "-fach"
        adverbial_suffixes = pynini.union("mal", "fach")
        graph_adverbials = (
            (graph_cardinal_canonical | graph_decimal)
            + (pynutil.delete("-") | pynutil.delete(NEMO_SPACE)).ques
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("morphosyntactic_features: ")
            + pynutil.insert('"')
            + adverbial_suffixes
            + pynutil.insert('"')
        )

        graphs_combined = graph_cardinal | graph_decimal | graph_fraction
        final_graph = (
            (graphs_combined + pynini.accep(NEMO_SPACE) + graph_unit)
            | graph_decimal_scrambled
            | graph_numeral_nouns
            | graph_adverbials
        )

        # Removes empty integer string
        removes_empty_ints = pynutil.delete('cardinal { integer: "" } ')
        remove_empty_ints = pynini.cdrewrite(removes_empty_ints, "", "", NEMO_SIGMA)
        final_graph = final_graph @ remove_empty_ints

        # Combines and optimizes everything
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
