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
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money (pt-BR), e.g.
        R$ 12 -> money { currency_maj: "reais" integer_part: "doze" }
        R$ 12,05 -> money { ... fractional_part: "cinco" currency_min: "centavos" preserve_order: true }
        R$ 0,20 -> money { fractional_part: "vinte" currency_min: "centavos" preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        decimal_separator = pynini.accep(",")
        maj_singular = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        maj_plural_map = pynini.string_file(get_abs_path("data/money/currency_major_plural.tsv"))
        maj_plural_graph = maj_singular @ maj_plural_map
        min_singular = pynini.string_file(get_abs_path("data/money/currency_minor.tsv"))
        min_plural_map = pynini.string_file(get_abs_path("data/money/currency_minor_plural.tsv"))
        min_plural_graph = min_singular @ min_plural_map

        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.final_graph_wo_negative

        graph_maj_singular = pynutil.insert('currency_maj: "') + maj_singular + pynutil.insert('"')
        graph_maj_plural = pynutil.insert('currency_maj: "') + maj_plural_graph + pynutil.insert('"')

        graph_integer_one = (
            pynutil.insert('integer_part: "') + (pynini.accep("1") @ cardinal_graph) + pynutil.insert('"')
        )

        decimal_with_quantity = (NEMO_SIGMA + NEMO_ALPHA) @ graph_decimal_final

        graph_decimal_plural = pynini.union(
            graph_maj_plural + pynini.closure(delete_space, 0, 1) + insert_space + graph_decimal_final,
            graph_decimal_final + pynini.closure(delete_space, 0, 1) + insert_space + graph_maj_plural,
        )
        graph_decimal_plural = ((NEMO_SIGMA - "1") + decimal_separator + NEMO_SIGMA) @ graph_decimal_plural

        graph_decimal_singular = pynini.union(
            graph_maj_singular + pynini.closure(delete_space, 0, 1) + insert_space + graph_decimal_final,
            graph_decimal_final + pynini.closure(delete_space, 0, 1) + insert_space + graph_maj_singular,
        )
        graph_decimal_singular = (pynini.accep("1") + decimal_separator + NEMO_SIGMA) @ graph_decimal_singular

        graph_decimal = pynini.union(
            graph_decimal_singular,
            graph_decimal_plural,
            graph_maj_plural + pynini.closure(delete_space, 0, 1) + insert_space + decimal_with_quantity,
        )

        graph_integer = pynutil.insert('integer_part: "') + ((NEMO_SIGMA - "1") @ cardinal_graph) + pynutil.insert('"')

        graph_integer_only = pynini.union(
            graph_maj_singular + pynini.closure(delete_space, 0, 1) + insert_space + graph_integer_one,
            graph_integer_one + pynini.closure(delete_space, 0, 1) + insert_space + graph_maj_singular,
        )
        graph_integer_only |= pynini.union(
            graph_maj_plural + pynini.closure(delete_space, 0, 1) + insert_space + graph_integer,
            graph_integer + pynini.closure(delete_space, 0, 1) + insert_space + graph_maj_plural,
        )

        graph = graph_integer_only | graph_decimal

        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )

        graph_min_singular = pynutil.insert('currency_min: "') + min_singular + pynutil.insert('"')
        graph_min_plural = pynutil.insert('currency_min: "') + min_plural_graph + pynutil.insert('"')

        maj_singular_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))
        decimal_graph_with_minor = None
        for curr_symbol, _ in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: true")

            integer_plus_maj = pynini.union(
                graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_plural,
                graph_integer_one + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular,
            )
            integer_plus_maj = (pynini.closure(NEMO_DIGIT) - "0") @ integer_plus_maj

            graph_fractional_one = (
                pynutil.insert('fractional_part: "')
                + (two_digits_fractional_part @ pynini.cross("1", "um"))
                + pynutil.insert('"')
            )

            graph_fractional = (
                two_digits_fractional_part @ (pynini.closure(NEMO_DIGIT, 1, 2) - "1") @ cardinal.two_digit_non_zero
            )
            graph_fractional = pynutil.insert('fractional_part: "') + graph_fractional + pynutil.insert('"')

            fractional_plus_min = pynini.union(
                graph_fractional + insert_space + pynutil.insert(curr_symbol) @ graph_min_plural,
                graph_fractional_one + insert_space + pynutil.insert(curr_symbol) @ graph_min_singular,
            )

            decimal_graph_with_minor_curr = (
                integer_plus_maj + pynini.cross(decimal_separator, NEMO_SPACE) + fractional_plus_min
            )
            if not deterministic:
                decimal_graph_with_minor_curr |= pynutil.add_weight(
                    integer_plus_maj
                    + pynini.cross(decimal_separator, NEMO_SPACE)
                    + pynutil.insert('fractional_part: "')
                    + two_digits_fractional_part @ cardinal.two_digit_non_zero
                    + pynutil.insert('"'),
                    weight=0.0001,
                )

            decimal_graph_with_minor_curr |= pynutil.delete("0,") + fractional_plus_min
            decimal_graph_with_minor_curr = pynini.union(
                pynutil.delete(curr_symbol)
                + pynini.closure(delete_space, 0, 1)
                + decimal_graph_with_minor_curr
                + preserve_order,
                decimal_graph_with_minor_curr
                + preserve_order
                + pynini.closure(delete_space, 0, 1)
                + pynutil.delete(curr_symbol),
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        final_graph = graph | pynutil.add_weight(decimal_graph_with_minor, -0.001)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
