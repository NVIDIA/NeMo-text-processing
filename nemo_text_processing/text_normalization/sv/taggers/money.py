# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.sv.graph_utils import SV_ALPHA, ensure_space
from nemo_text_processing.text_normalization.sv.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        $12,05 -> money { integer_part: "tolv" currency_maj: "dollar" fractional_part: "fem" currency_min: "cent" preserve_order: true }
        $12,0500 -> money { integer_part: "tolv" currency_maj: "dollar" fractional_part: "fem" currency_min: "cent" preserve_order: true }
        $1 -> money { currency_maj: "dollar" integer_part: "en" }
        $1,00 -> money { currency_maj: "dollar" integer_part: "en" }
        $0,05 -> money { fractional_part: "fem"  currency_min: "cent" preserve_order: true }
        $1 miljon -> money { currency_maj: "dollar" integer_part: "en" quantity: "miljon" }
        $1,2 miljon -> money { currency_maj: "dollar" integer_part: "en"  fractional_part: "två" quantity: "miljon" }
        $1,2320 -> money { currency_maj: "dollar" integer_part: "en"  fractional_part: "two three two" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph_no_one_en
        cardinal_graph_ett = cardinal.graph_no_one
        graph_decimal_final = decimal.final_graph_wo_negative_w_abbr_en
        graph_decimal_final_ett = decimal.final_graph_wo_negative_w_abbr

        min_singular = pynini.string_file(get_abs_path("data/money/currency_minor_singular.tsv"))
        min_plural = pynini.string_file(get_abs_path("data/money/currency_minor_plural.tsv"))
        maj_singular = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        maj_singular_nt = pynini.string_file(get_abs_path("data/money/currency_major_nt.tsv"))
        maj_plural = pynini.string_file(get_abs_path("data/money/currency_plurals.tsv"))

        maj_singular_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))
        maj_singular_labels_nt = load_labels(get_abs_path("data/money/currency_major_nt.tsv"))
        maj_unit_plural = convert_space(maj_singular @ maj_plural)
        maj_unit_plural_nt = convert_space(maj_singular_nt @ maj_plural)
        maj_unit_singular = convert_space(maj_singular)
        maj_unit_singular_nt = convert_space(maj_singular_nt)

        self.maj_unit_plural = maj_unit_plural
        self.maj_unit_plural_nt = maj_unit_plural_nt
        self.maj_unit_singular = maj_unit_singular
        self.maj_unit_singular_nt = maj_unit_singular_nt

        graph_maj_singular = pynutil.insert("currency_maj: \"") + maj_unit_singular + pynutil.insert("\"")
        graph_maj_plural = pynutil.insert("currency_maj: \"") + maj_unit_plural + pynutil.insert("\"")
        graph_maj_singular_nt = pynutil.insert("currency_maj: \"") + maj_unit_singular_nt + pynutil.insert("\"")
        graph_maj_plural_nt = pynutil.insert("currency_maj: \"") + maj_unit_plural_nt + pynutil.insert("\"")

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(",") + pynini.closure(pynutil.delete("0"), 1), 0, 1
        )

        graph_integer_sg_en = pynutil.insert("integer_part: \"") + pynini.cross("1", "en") + pynutil.insert("\"")
        graph_integer_sg_ett = pynutil.insert("integer_part: \"") + pynini.cross("1", "ett") + pynutil.insert("\"")
        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(NEMO_DIGIT | pynutil.delete(" "))
            + pynini.accep(",")
            + pynini.closure(NEMO_DIGIT, 2)
            + (NEMO_DIGIT - "0")
            + pynini.closure(pynutil.delete("0"))
        )
        decimal_with_quantity = NEMO_SIGMA + SV_ALPHA

        decimal_part = (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final
        decimal_part_ett = (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final_ett
        graph_decimal = pynini.union(
            graph_maj_plural + ensure_space + decimal_part,
            graph_maj_plural_nt + ensure_space + decimal_part_ett,
            decimal_part_ett + ensure_space + graph_maj_plural_nt,
            decimal_part + ensure_space + graph_maj_plural,
        )

        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer_ett = pynutil.insert("integer_part: \"") + cardinal_graph_ett + pynutil.insert("\"")

        graph_integer_only = graph_maj_singular + ensure_space + graph_integer_sg_en
        graph_integer_only |= graph_maj_singular_nt + ensure_space + graph_integer_sg_ett
        graph_integer_only |= graph_maj_plural + ensure_space + graph_integer
        graph_integer_only |= graph_maj_plural_nt + ensure_space + graph_integer_ett
        graph_integer_only |= graph_integer_sg_en + ensure_space + graph_maj_singular
        graph_integer_only |= graph_integer_sg_ett + ensure_space + graph_maj_singular_nt
        graph_integer_only |= graph_integer + ensure_space + graph_maj_plural
        graph_integer_only |= graph_integer_ett + ensure_space + graph_maj_plural_nt

        final_graph = (graph_integer_only + optional_delete_fractional_zeros) | graph_decimal

        # remove trailing zeros of non zero number in the first 2 digits and fill up to 2 digits
        # e.g. 2000 -> 20, 0200->02, 01 -> 01, 10 -> 10
        # not accepted: 002, 00, 0,
        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )

        graph_min_singular = pynutil.insert(" currency_min: \"") + min_singular + pynutil.insert("\"")
        graph_min_plural = pynutil.insert(" currency_min: \"") + min_plural + pynutil.insert("\"")

        maj_singular_labels_all = [(x[0], "en") for x in maj_singular_labels]
        maj_singular_labels_all += [(x[0], "ett") for x in maj_singular_labels_nt]
        # format ** dollars ** cent
        decimal_graph_with_minor = None
        integer_graph_reordered = None
        decimal_default_reordered = None
        for curr_symbol, one_form in maj_singular_labels_all:
            preserve_order = pynutil.insert(" preserve_order: true")
            if one_form == "en":
                integer_plus_maj = graph_integer + insert_space + (pynutil.insert(curr_symbol) @ graph_maj_plural)
                integer_plus_maj |= (
                    graph_integer_sg_en + insert_space + (pynutil.insert(curr_symbol) @ graph_maj_singular)
                )
            else:
                integer_plus_maj = (
                    graph_integer_ett + insert_space + (pynutil.insert(curr_symbol) @ graph_maj_plural_nt)
                )
                integer_plus_maj |= (
                    graph_integer_sg_ett + insert_space + (pynutil.insert(curr_symbol) @ graph_maj_singular_nt)
                )

            integer_plus_maj_with_comma = pynini.compose(
                NEMO_DIGIT - "0" + pynini.closure(NEMO_DIGIT | delete_space), integer_plus_maj
            )
            integer_plus_maj = pynini.compose(pynini.closure(NEMO_DIGIT) - "0", integer_plus_maj)
            integer_plus_maj |= integer_plus_maj_with_comma

            # all of the minor currency units are "en"
            graph_fractional_one = two_digits_fractional_part @ pynini.cross("1", "en")
            graph_fractional_one = pynutil.insert("fractional_part: \"") + graph_fractional_one + pynutil.insert("\"")
            graph_fractional = (
                two_digits_fractional_part
                @ (pynini.closure(NEMO_DIGIT, 1, 2) - "1")
                @ cardinal.graph_hundreds_component_at_least_one_non_zero_digit_en
            )
            graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional + pynutil.insert("\"")

            fractional_plus_min = graph_fractional + ensure_space + (pynutil.insert(curr_symbol) @ graph_min_plural)
            fractional_plus_min |= (
                graph_fractional_one + ensure_space + (pynutil.insert(curr_symbol) @ graph_min_singular)
            )

            decimal_graph_with_minor_curr = integer_plus_maj + pynini.cross(",", " ") + fractional_plus_min

            if not deterministic:
                decimal_graph_with_minor_curr |= pynutil.add_weight(
                    integer_plus_maj
                    + pynini.cross(",", " ")
                    + pynutil.insert("fractional_part: \"")
                    + two_digits_fractional_part @ cardinal.graph_hundreds_component_at_least_one_non_zero_digit_en
                    + pynutil.insert("\""),
                    weight=0.0001,
                )
                default_fraction_graph = (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final

            decimal_graph_with_minor_curr |= (
                pynini.closure(pynutil.delete("0"), 0, 1) + pynutil.delete(",") + fractional_plus_min
            )
            decimal_graph_with_minor_curr = (
                pynutil.delete(curr_symbol) + decimal_graph_with_minor_curr + preserve_order
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr).optimize()
            )

            if not deterministic:
                integer_graph_reordered_curr = (
                    pynutil.delete(curr_symbol) + integer_plus_maj + preserve_order
                ).optimize()

                integer_graph_reordered = (
                    integer_graph_reordered_curr
                    if integer_graph_reordered is None
                    else pynini.union(integer_graph_reordered, integer_graph_reordered_curr).optimize()
                )
                decimal_default_reordered_curr = (
                    pynutil.delete(curr_symbol)
                    + default_fraction_graph
                    + ensure_space
                    + pynutil.insert(curr_symbol) @ graph_maj_plural
                )

                decimal_default_reordered = (
                    decimal_default_reordered_curr
                    if decimal_default_reordered is None
                    else pynini.union(decimal_default_reordered, decimal_default_reordered_curr)
                ).optimize()

        # weight for SH
        final_graph |= pynutil.add_weight(decimal_graph_with_minor, -0.0001)

        if not deterministic:
            final_graph |= integer_graph_reordered | decimal_default_reordered
            # to handle "$2.00" cases
            final_graph |= pynini.compose(
                NEMO_SIGMA + pynutil.delete(",") + pynini.closure(pynutil.delete("0"), 1), integer_graph_reordered
            )
        final_graph = self.add_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
