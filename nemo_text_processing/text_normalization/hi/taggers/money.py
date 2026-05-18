# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path, load_labels

currency_graph          = pynini.string_file(get_abs_path("data/money/currency.tsv"))
currency_singular_graph = pynini.string_file(get_abs_path("data/money/currency_singular.tsv"))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹५० -> money { currency_maj: "रुपए" integer_part: "पचास" }
        ₹५०.५० -> money { currency_maj: "रुपए" integer_part: "पचास" fractional_part: "पचास" currency_min: "पैसे" }
        ₹०.५० -> { money { currency_maj: "रुपए" integer_part: "शून्य" fractional_part: "पचास" currency_min: "पैसे" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.final_graph

        _en_to_hi_digit = pynini.string_file(get_abs_path("data/ordinal/en_to_hi_digit.tsv"))
        _deva_to_ascii  = pynini.invert(_en_to_hi_digit)
        deva_to_ascii   = pynini.closure(_deva_to_ascii | pynini.union(*"0123456789"), 1)

        _ascii_digit   = pynini.union(*"0123456789")
        _ascii_nonzero = pynini.union(*"123456789")
        _deva_nonzero  = pynini.union(*"१२३४५६७८९")
        _any_digit     = _ascii_digit | pynini.union(*"०१२३४५६७८९")
        _any_nonzero   = _ascii_nonzero | _deva_nonzero

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true"') + insert_space,
            0,
            1,
        )

        currency_major = (
            pynutil.insert('currency_maj: "') + currency_graph + pynutil.insert('"')
        )
        currency_major_singular = (
            pynutil.insert('currency_maj: "') + currency_singular_graph + pynutil.insert('"')
        )

        one = pynini.union("1", "१")
        integer_one = (
            pynutil.insert('integer_part: "') + (one @ cardinal_graph) + pynutil.insert('"')
        )
        integer = (
            pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        )

        strip_trailing_zeros = (
            pynini.closure(_ascii_digit) + _ascii_nonzero + pynini.closure(pynutil.delete("0"))
        )
        canonicalise = (
            (pynutil.delete("0") + _ascii_nonzero)
            | (_ascii_nonzero + pynutil.insert("0"))
            | (_ascii_nonzero + _ascii_digit)
        )
        two_digits_fractional_part = deva_to_ascii @ strip_trailing_zeros @ canonicalise

        fraction = (
            pynutil.insert('fractional_part: "')
            + (two_digits_fractional_part @ cardinal_graph)
            + pynutil.insert('"')
        )

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(".")
            + pynini.closure(pynutil.delete("0") | pynutil.delete("०"), 1),
            0,
            1,
        )

        has_3plus_sig_digits = (
            _any_digit + _any_digit + _any_nonzero + pynini.closure(_any_digit)
        )
        single_digit   = _any_digit @ cardinal.single_digits_graph
        decimal_digits = (
            pynutil.insert('fractional_part: "')
            + single_digit
            + pynini.closure(insert_space + single_digit)
            + pynutil.insert('"')
        )
        guarded_decimal_digits = has_3plus_sig_digits @ decimal_digits

        graph_decimal_path = (
            optional_graph_negative
            + currency_major
            + insert_space
            + pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynini.cross(".", " ")
            + guarded_decimal_digits
        ).optimize()

        graph_major_only_singular = (
            optional_graph_negative
            + currency_major_singular
            + insert_space
            + integer_one
            + optional_delete_fractional_zeros
        ).optimize()

        graph_major_only = (
            optional_graph_negative
            + currency_major
            + insert_space
            + integer
            + optional_delete_fractional_zeros
        ).optimize()

        maj_labels          = load_labels(get_abs_path("data/money/currency.tsv"))
        maj_singular_labels = load_labels(get_abs_path("data/money/currency_singular.tsv"))
        maj_to_min          = dict(load_labels(get_abs_path("data/money/major_minor_currencies.tsv")))

        def _build_major_and_minor(sym_maj_labels, int_graph):
            result = None
            for sym, maj in sym_maj_labels:
                min_name = maj_to_min.get(maj)
                if not min_name:
                    continue

                curr_maj = (
                    pynutil.insert('currency_maj: "')
                    + pynini.cross(sym, maj)   
                    + pynutil.insert('"')
                )
                curr_min = (
                    pynutil.insert('currency_min: "')
                    + pynutil.insert(min_name) 
                    + pynutil.insert('"')
                )

                g = (
                    optional_graph_negative
                    + curr_maj
                    + insert_space
                    + int_graph
                    + pynini.cross(".", " ")
                    + fraction
                    + insert_space
                    + curr_min
                ).optimize()

                result = g if result is None else pynini.union(result, g).optimize()

            return result

        graph_major_and_minor          = _build_major_and_minor(maj_labels,          integer)
        graph_major_and_minor_singular = _build_major_and_minor(maj_singular_labels, integer_one)

        graph_currencies = (
            pynutil.add_weight(
                graph_major_only_singular | graph_major_and_minor_singular, -0.001
            )
            | pynutil.add_weight(graph_decimal_path, -0.0005)
            | graph_major_only
            | graph_major_and_minor
        )

        graph = graph_currencies.optimize()
        self.fst = self.add_tokens(graph)