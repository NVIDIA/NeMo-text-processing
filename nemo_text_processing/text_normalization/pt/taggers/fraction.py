# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_WHITE_SPACE,
    insert_space,
    GraphFst,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese fraction numbers, e.g.
        "1/2" -> fraction { numerator: "um" denominator: "meio" morphosyntactic_features: "ordinal" }
        "2 3/4" -> fraction { integer_part: "dois" numerator: "três" denominator: "quarto" ... }
        "2/11" -> fraction { numerator: "dois" denominator: "onze" morphosyntactic_features: "avos" }

    Args:
        cardinal: CardinalFst instance for number parts.
        ordinal: OrdinalFst instance for denominator 2-10 and exceptions.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        # Denominators 2–10 use ordinal form (no data file: fixed set)
        two_to_ten = pynini.union(
            *[pynini.accep(str(d)) for d in range(2, 11)]
        ).optimize()

        ord_digit_rows = load_labels(get_abs_path("data/ordinals/digit.tsv"))
        ordinal_digit = pynini.string_map(
            [(r[1], r[0]) for r in ord_digit_rows if len(r) >= 2]
        ).optimize()

        ord_exc_rows = load_labels(get_abs_path("data/fractions/ordinal_exceptions.tsv"))
        ordinal_exceptions = pynini.string_map(
            [(r[0], r[1]) for r in ord_exc_rows if len(r) >= 2]
        ).optimize()

        ord_hundreds_rows = load_labels(get_abs_path("data/ordinals/hundreds.tsv"))
        ordinal_hundreds = pynini.string_map(
            [(r[1], r[0]) for r in ord_hundreds_rows if len(r) >= 2]
        ).optimize()

        powers_rows = load_labels(get_abs_path("data/fractions/powers_of_ten.tsv"))
        powers_of_ten = pynini.string_map(
            [(r[0], r[1]) for r in powers_rows if len(r) >= 2]
        ).optimize()

        denom_ordinal_form = two_to_ten @ cardinal_graph @ ordinal_digit
        denom_ordinal_form = denom_ordinal_form @ pynini.cdrewrite(
            ordinal_exceptions, "", "", NEMO_SIGMA
        )
        denom_ordinal = (
            pynutil.insert('denominator: "')
            + denom_ordinal_form
            + pynutil.insert('" morphosyntactic_features: "ordinal"')
        )

        denom_100 = (
            pynutil.insert('denominator: "')
            + (pynini.accep("100") @ cardinal_graph @ ordinal_hundreds)
            + pynutil.insert('" morphosyntactic_features: "ordinal"')
        )
        denom_1000 = (
            pynutil.insert('denominator: "')
            + (pynini.accep("1000") @ cardinal_graph @ powers_of_ten)
            + pynutil.insert('" morphosyntactic_features: "ordinal"')
        )

        denom_ordinal_2_10_100_1000 = pynini.union(
            denom_ordinal, denom_100, denom_1000
        )
        digit_plus = pynini.closure(NEMO_DIGIT, 1)
        denom_avos_input = pynini.difference(
            digit_plus,
            pynini.union(
                two_to_ten,
                pynini.accep("100"),
                pynini.accep("1000"),
            ),
        )
        denom_avos = (
            pynutil.insert('denominator: "')
            + (denom_avos_input @ cardinal_graph)
            + pynutil.insert('" morphosyntactic_features: "avos"')
        )

        denominator = pynini.union(denom_ordinal_2_10_100_1000, denom_avos)

        # Slash variants: ASCII /, Unicode ⁄ (U+2044), ∕ (U+2215); with or without spaces
        slash_or_space_slash = pynini.union(
            pynini.cross("/", '" '),
            pynini.cross(" / ", '" '),
            pynini.cross("\u2044", '" '),   # fraction slash ⁄
            pynini.cross(" \u2044 ", '" '),
            pynini.cross("\u2215", '" '),   # division slash ∕
            pynini.cross(" \u2215 ", '" '),
        )
        numerator = (
            pynutil.insert('numerator: "')
            + cardinal_graph
            + slash_or_space_slash
        )
        fraction_core = numerator + denominator

        integer_part = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
            + insert_space
        )

        optional_minus = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        mixed = (
            integer_part
            + pynini.closure(NEMO_WHITE_SPACE, 1)
            + fraction_core
        )
        graph = optional_minus + pynini.union(mixed, fraction_core)

        self.fst = self.add_tokens(graph).optimize()
