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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measures:
        e.g. measure { cardinal { integer: "2" } units: "kg" } -> 2 kg
        e.g. measure { cardinal { negative: "-" integer: "67" } units: "kg" } -> -67 kg
        e.g. measure { decimal { integer_part: "1" fractional_part: "5" } units: "km/h" } -> 1,5 km/h
        e.g. measure { fraction { 3/4 } units: "ω" } -> 3/4 ω
        e.g. measure { integer_part: "1" units: "m" fractional_part: "50" } -> 1,50 m
        e.g. measure { cardinal { integer: "16" } morphosyntactic_features: "er" } -> 16er
        e.g. measure { cardinal { integer: "14" } morphosyntactic_features: "fach" } -> 14-fach
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst):
        super().__init__(name="measure", kind="verbalize")
        graph_cardinals = cardinal.numbers
        graph_cardinals_first_dozen = cardinal.first_dozen
        graph_decimals = decimal.numbers
        graph_fractions = fraction.numbers
        graph_negative = pynini.cross('negative: "-"', "-") + pynutil.delete(NEMO_SPACE)
        numeral_noun_suffixes = pynini.string_file(get_abs_path("data/measure/numeral_noun_suffixes.tsv"))
        adverbial_suffixes = pynini.string_file(get_abs_path("data/measure/adverbial_suffixes.tsv"))

        graph_unit = (
            pynutil.delete("units:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + (pynini.closure(NEMO_NOT_QUOTE, 1) | pynini.closure(pynini.difference(NEMO_CHAR, NEMO_SPACE), 1))
            + pynutil.delete('"')
        )

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
            graph_negative.ques
            + pynutil.delete('integer_part: "')
            + pynini.closure(cardinal_component, 1)
            + pynutil.delete('"')
            + pynini.cross(NEMO_SPACE, ",")
            + pynutil.delete('fractional_part: "')
            + pynini.closure(NEMO_DIGIT, 1)
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
        graph_numeral_noun = (
            (graph_cardinal_first_dozen | graph_cardinal)
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("morphosyntactic_features:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + numeral_noun_suffixes
            + pynutil.delete('"')
        )

        # Handles numeral adverbials created with affixed "-mal" or "-fach"
        # The block below handles both hyphenated and non-hyphenated cases
        graph_suffixes = (
            pynutil.delete("morphosyntactic_features: ")
            + pynutil.delete('"')
            + adverbial_suffixes
            + pynutil.delete('"')
        )

        graph_hyphenated = (
            (graph_cardinal | graph_decimal | graph_fraction) + pynini.cross(NEMO_SPACE, "-") + graph_suffixes
        )

        graph_non_hyphenated = graph_cardinal_first_dozen + pynutil.delete(NEMO_SPACE).ques + graph_suffixes

        graph_adverbials = (graph_hyphenated | pynutil.add_weight(graph_non_hyphenated, 0.01)).optimize()

        aggregated_suffixes = numeral_noun_suffixes | adverbial_suffixes
        graph_morphosyntactic_features = (
            pynutil.delete("morphosyntactic_features: ")
            + pynutil.delete('"')
            + aggregated_suffixes
            + pynutil.delete('"')
        )

        graph = (
            (
                (graph_decimal | graph_cardinal | graph_fraction | graph_scrambled_decimal)
                + pynini.accep(NEMO_SPACE)
                + graph_unit
            )
            | graph_numeral_noun
            | graph_adverbials
            | graph_morphosyntactic_features
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
