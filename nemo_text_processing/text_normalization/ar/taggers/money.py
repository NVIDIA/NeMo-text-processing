# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path, load_labels

min_singular = pynini.string_file(get_abs_path("data/money/currency_minor_singular.tsv"))
min_plural = pynini.string_file(get_abs_path("data/money/currency_minor_plural.tsv"))
maj_singular = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
ar_cur = pynini.string_file((get_abs_path("data/money/local_currency.tsv")))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "$1,99" -> money { integer_part: "سبعة" currency_maj: "دولار" fractional_part: "تسعة وتسعون"  currency_min: "سنت" preserve_order: true}
        "$0,10" -> money { fractional_part: "عشرة"  currency_min: "بنسات" preserve_order: true }
        "$9" -> money { integer_part: "تسعة" currency_maj: "دولار" preserve_order: true}

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers_with_leading_zeros
        # graph_decimal_final = final_graph_decimal

        maj_singular_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))
        maj_singular_graph = convert_space(maj_singular)
        maj_plural_graph = maj_singular_graph
        ar_cur_graph = convert_space(ar_cur)

        graph_maj_singular = pynutil.insert("currency_maj: \"") + maj_singular_graph + pynutil.insert("\"")
        graph_maj_plural = pynutil.insert("currency_maj: \"") + maj_plural_graph + pynutil.insert("\"")
        graph_ar_cur = pynutil.insert("currency_maj: \"") + ar_cur_graph + pynutil.insert("\"")

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(".") + pynini.closure(pynutil.delete("0"), 1), 0, 1
        )
        graph_integer_one = pynutil.insert("integer_part: \"") + pynini.cross("1", "واحد") + pynutil.insert("\"")

        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(NEMO_DIGIT, 1)
            + pynini.accep(".")
            + pynini.closure(NEMO_DIGIT, 2)
            + (NEMO_DIGIT - "0")
            + pynini.closure(pynutil.delete("0"))
        )
        decimal_with_quantity = NEMO_SIGMA + NEMO_ALPHA
        # graph_decimal = (
        # graph_maj_plural + insert_space + (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final
        # )

        graph_integer = (
            pynutil.insert("integer_part: \"") + ((NEMO_SIGMA - "1") @ cardinal_graph) + pynutil.insert("\"")
        )

        graph_integer_only = graph_maj_singular + insert_space + graph_integer_one
        graph_integer_only |= graph_maj_plural + insert_space + graph_integer

        # For local currency "9د.ك"
        graph_integer_only_ar = graph_integer + insert_space + graph_ar_cur
        # graph_decimal_ar = graph_decimal_final + insert_space  + graph_ar_cur

        graph = (graph_integer_only + optional_delete_fractional_zeros) | graph_integer_only_ar

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

        # format ** euro ** cent

        decimal_graph_with_minor = None
        graph_with_no_minor = None

        for curr_symbol, _ in maj_singular_labels:

            preserve_order = pynutil.insert(" preserve_order: true")
            integer_plus_maj = graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_plural
            integer_plus_maj |= graph_integer_one + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular
            # non zero integer part
            integer_plus_maj = (pynini.closure(NEMO_DIGIT) - "0") @ integer_plus_maj

            graph_fractional_one = two_digits_fractional_part @ pynini.cross("1", "")
            graph_fractional_one = pynutil.insert("fractional_part: \"") + graph_fractional_one + pynutil.insert("\"")

            digits_two_to_ten = pynini.union("2", "3", "4", "5", "6", "7", "8", "9", "10")

            graph_fractional_up_to_ten = two_digits_fractional_part @ digits_two_to_ten @ cardinal_graph
            graph_fractional_up_to_ten = (
                pynutil.insert("fractional_part: \"") + graph_fractional_up_to_ten + pynutil.insert("\"")
            )

            graph_fractional = (
                two_digits_fractional_part
                @ (pynini.closure(NEMO_DIGIT, 1, 2) - pynini.union("2", "3", "4", "5", "6", "7", "8", "9", "10"))
                @ cardinal_graph
            )
            graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional + pynutil.insert("\"")

            fractional_plus_min = graph_fractional + insert_space + pynutil.insert(curr_symbol) @ graph_min_singular
            fractional_plus_min |= (
                graph_fractional_one + insert_space + pynutil.insert(curr_symbol) @ graph_min_singular
            )
            fractional_plus_min |= (
                graph_fractional_up_to_ten + insert_space + pynutil.insert(curr_symbol) @ graph_min_plural
            )

            graph_with_no_minor_curr = integer_plus_maj
            graph_with_no_minor_curr |= pynutil.add_weight(integer_plus_maj, weight=0.0001,)

            graph_with_no_minor_curr = pynutil.delete(curr_symbol) + graph_with_no_minor_curr + preserve_order

            graph_with_no_minor = (
                graph_with_no_minor_curr
                if graph_with_no_minor is None
                else pynini.union(graph_with_no_minor, graph_with_no_minor_curr)
            )
            decimal_graph_with_minor_curr = integer_plus_maj + pynini.cross(".", " ") + fractional_plus_min
            decimal_graph_with_minor_curr |= pynutil.add_weight(
                integer_plus_maj
                + pynini.cross(".", " ")
                + pynutil.insert("fractional_part: \"")
                + two_digits_fractional_part @ cardinal_graph
                + pynutil.insert("\""),
                weight=0.0001,
            )
            decimal_graph_with_minor_curr |= pynutil.delete("0.") + fractional_plus_min
            decimal_graph_with_minor_curr = (
                pynutil.delete(curr_symbol) + decimal_graph_with_minor_curr + preserve_order
            )
            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        final_graph = decimal_graph_with_minor | graph_with_no_minor | graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
