# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from nemo_text_processing.text_normalization.ja.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying Japanese money expressions.

    Examples:
        100円 -> money { integer_part: "百" currency_maj: "円" preserve_order: true }
        ¥3万 -> money { integer_part: "三" quantity: "万" currency_maj: "円" preserve_order: true }
        1.5万円 -> money { integer_part: "一点五" quantity: "万" currency_maj: "円" preserve_order: true }
        5ドル25セント -> money { integer_part: "五" currency_maj: "ドル" fractional_part: "二十五" currency_min: "セント" preserve_order: true }
        $12.50 -> money { integer_part: "十二" currency_maj: "ドル" fractional_part: "五十" currency_min: "セント" preserve_order: true }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.just_cardinals
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        integer_input = (NEMO_DIGIT + pynini.closure(NEMO_DIGIT | pynutil.delete(","))) @ graph_cardinal
        fractional_digits = pynini.closure(graph_digit | graph_zero, 1)
        decimal_input = integer_input + pynutil.delete(".") + pynutil.insert("点") + fractional_digits
        sign_input = (pynini.cross("-", "マイナス") | pynini.accep("マイナス")) + delete_space

        integer_component = pynutil.insert('integer_part: "') + integer_input + pynutil.insert('"')
        decimal_component = pynutil.insert('integer_part: "') + decimal_input + pynutil.insert('"')
        signed_integer_component = (
            pynutil.insert('integer_part: "') + pynini.closure(sign_input, 0, 1) + integer_input + pynutil.insert('"')
        )
        signed_decimal_component = (
            pynutil.insert('integer_part: "') + pynini.closure(sign_input, 0, 1) + decimal_input + pynutil.insert('"')
        )

        number_component = signed_decimal_component | signed_integer_component

        quantity = pynini.string_file(get_abs_path("data/money/quantity.tsv"))
        quantity_component = delete_space + pynutil.insert(' quantity: "') + quantity + pynutil.insert('"')

        currency_major_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))
        currency_major = pynini.string_file(get_abs_path("data/money/currency_major.tsv"))
        currency_major_component = (
            delete_space + pynutil.insert(' currency_maj: "') + currency_major + pynutil.insert('"')
        )

        currency_prefix_labels = load_labels(get_abs_path("data/money/currency_prefix.tsv"))
        currency_minor_by_major = dict(load_labels(get_abs_path("data/money/currency_minor_by_major.tsv")))
        currency_minor = pynini.string_file(get_abs_path("data/money/currency_minor.tsv"))
        non_zero_digit = pynini.difference(NEMO_DIGIT, "0")
        minor_decimal_input = (
            (NEMO_DIGIT**2 @ graph_cardinal) | (pynutil.delete("0") + (non_zero_digit @ graph_digit))
        )

        suffix_graph = (
            number_component
            + pynini.closure(quantity_component, 0, 1)
            + currency_major_component
        )
        for written, spoken in currency_major_labels:
            minor_spoken = currency_minor_by_major.get(written)
            if not minor_spoken:
                continue

            currency_major_suffix = (
                delete_space
                + pynutil.delete(written)
                + pynutil.insert(f' currency_maj: "{spoken}"')
            )
            minor_suffix = (
                delete_space
                + pynutil.insert(' fractional_part: "')
                + integer_input
                + pynutil.insert('"')
                + delete_space
                + (currency_minor @ pynini.cross(minor_spoken, ""))
                + pynutil.insert(f' currency_min: "{minor_spoken}"')
            )
            suffix_graph |= (
                signed_integer_component
                + currency_major_suffix
                + minor_suffix
            )

        prefix_graph = pynini.Fst()
        for written, spoken in currency_prefix_labels:
            currency_prefix = pynutil.delete(written) + delete_space
            currency_field = pynutil.insert(f' currency_maj: "{spoken}"')
            minor_spoken = currency_minor_by_major.get(written)

            prefix_graph |= (
                currency_prefix
                + number_component
                + pynini.closure(quantity_component, 0, 1)
                + currency_field
            )
            prefix_graph |= (
                pynutil.insert('integer_part: "')
                + sign_input
                + currency_prefix
                + (decimal_input | integer_input)
                + pynutil.insert('"')
                + pynini.closure(quantity_component, 0, 1)
                + currency_field
            )
            if minor_spoken:
                decimal_minor_component = (
                    integer_component
                    + pynutil.delete(".")
                    + currency_field
                    + pynutil.insert(' fractional_part: "')
                    + minor_decimal_input
                    + pynutil.insert(f'" currency_min: "{minor_spoken}"')
                )
                signed_decimal_minor_component = (
                    pynutil.insert('integer_part: "')
                    + sign_input
                    + currency_prefix
                    + integer_input
                    + pynutil.delete(".")
                    + pynutil.insert('"')
                    + currency_field
                    + pynutil.insert(' fractional_part: "')
                    + minor_decimal_input
                    + pynutil.insert(f'" currency_min: "{minor_spoken}"')
                )
                prefix_graph |= pynutil.add_weight(
                    currency_prefix + decimal_minor_component,
                    -0.1,
                )
                prefix_graph |= pynutil.add_weight(
                    signed_decimal_minor_component,
                    -0.1,
                )

        graph = (suffix_graph | prefix_graph) + pynutil.insert(" preserve_order: true")

        self.fst = self.add_tokens(graph.optimize()).optimize()
