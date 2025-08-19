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


from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_CHAR,
    NEMO_SPACE,
    NEMO_DIGIT,
    NEMO_ALPHA,
    GraphFst,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measures:
        e.g. measure { cardinal: "2" units: "kg" } -> 2 kg
        e.g. measure { decimal { integer_part: "1" fractional_part: "5" } units: "km/h" } -> 1,5 km/h
        e.g. measure { fraction: "3/4" units: "ω" } -> 3/4 ω
        e.g. measure { cardinal: "-2.000.000" units: "m²/s" } -> -2.000.000 m²/s
        e.g. measure { cardinal: "16" morphosyntactic_features: "er" } -> 16er
        e.g. measure { units: "Kapitel" cardinal: "4" } -> Kapitel 4
        e.g. measure { cardinal: "14" morphosyntactic_features: "fach" } -> 14-fach
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst):
        super().__init__(name="measure", kind="verbalize")
        graph_cardinals = cardinal.numbers
        graph_cardinals_first_dozen = cardinal.first_dozen
        graph_decimals = decimal.numbers
        graph_fractions = fraction.numbers
        graph_negative = pynini.cross('negative: "-"', "-") + pynutil.delete(NEMO_SPACE)
        graph_unit = (
            pynutil.delete("units:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.closure(pynini.difference(NEMO_CHAR, NEMO_SPACE), 1)
            + pynutil.delete('"')
        )

        # DE_chars = pynini.union(*"äöüÄÖÜß").optimize()
        cardinal_component = NEMO_DIGIT | pynini.accep(".") | pynini.accep("-")

        #  Handles the CARDINAL grammar
        graph_cardinal = (
            pynutil.delete("cardinal {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_cardinals
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        graph_cardinal_first_dozen = (
            pynutil.delete("cardinal {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_cardinals_first_dozen
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        # Handles the DECIMAL grammar
        graph_decimal = (
            pynutil.delete("decimal {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_decimals
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        # Handles scrambled decimals
        graph_scrambled_decimal = (
            graph_cardinal
            + pynini.cross(NEMO_SPACE, ",")
            + pynutil.delete("morphosyntactic_features:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.closure(cardinal_component, 1)
            + pynutil.delete('"')
        )

        # Handles the FRACTION grammar
        graph_fraction = (
            pynutil.delete("fraction {")
            + pynutil.delete(NEMO_SPACE)
            + graph_negative.ques
            + graph_fractions
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("}")
        )

        # Handles numeral nouns
        suffix = pynini.accep("er") + pynini.union(*"nmrs").ques
        graph_numeral_noun = (
            graph_cardinal_first_dozen
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("morphosyntactic_features:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + suffix
            + pynutil.delete('"')
        )

        # Handles numeral adverbials created with affixed "-mal" or "-fach"
        # The block below handles both hyphenated and non-hyphenated cases
        suffixes = pynini.union("mal", "fach")
        graph_suffixes = (
            pynutil.delete("morphosyntactic_features: ")
            + pynutil.delete('"')
            + suffixes
            + pynutil.delete('"')
        )

        graph_hyphenated = (
            (graph_cardinal | graph_decimal | graph_fraction)
            + pynini.cross(NEMO_SPACE, "-")
            + graph_suffixes
        )

        graph_non_hyphenated = (
            graph_cardinal_first_dozen
            + pynutil.delete(NEMO_SPACE).ques
            + graph_suffixes
        )

        graph_adverbials = (
            graph_hyphenated | pynutil.add_weight(graph_non_hyphenated, 0.01)
        ).optimize()

        # Handles only the morphosyntactic_features field
        aggregated_suffixes = suffix | pynini.accep("mal") | pynini.accep("fach")
        graph_morphosyntactic_features = (
            pynutil.delete("morphosyntactic_features: ")
            + pynutil.delete('"')
            + aggregated_suffixes
            + pynutil.delete('"')
        )

        graph = (
            (
                (
                    graph_decimal
                    | graph_cardinal
                    | graph_fraction
                    | graph_scrambled_decimal
                )
                + pynini.accep(NEMO_SPACE)
                + graph_unit
            )
            | graph_numeral_noun
            | graph_adverbials
            | graph_morphosyntactic_features
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
