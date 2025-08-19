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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    NEMO_DIGIT,
    NEMO_SPACE,
    NEMO_SIGMA,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money,
        e.g. money { integer_part: "0" fractional_part: "3" currency: "¢" } -> 3 ¢
        e.g. money { integer_part: "7" currency: "€" } -> 7 €
        e.g. money { integer_part: "76.987" currency: "$" } -> 76.987 $
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        integer_components = NEMO_DIGIT | "."
        currency_mappings = pynini.string_file(
            get_abs_path("data/money/currency_major.tsv")
        )
        get_powers_of_ten = pynini.string_file(
            get_abs_path("data/money/magnitudes.tsv")
        )
        currency_symbol = pynini.project(currency_mappings, "output")

        graph_major_only = (
            pynutil.delete('integer_part: "')
            + (
                pynini.closure(integer_components)
                @ pynini.cdrewrite(get_powers_of_ten, "", "[EOS]", NEMO_SIGMA)
                @ pynini.cdrewrite(
                    pynutil.insert(NEMO_SPACE),
                    NEMO_DIGIT,
                    pynini.project(get_powers_of_ten, "output"),
                    NEMO_SIGMA,
                )
            )
            + pynutil.delete('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.delete('currency: "')
            + currency_symbol
            + pynutil.delete('"')
        )

        # A separate graph for general-purpose cents
        cents = pynini.accep("¢")
        graph_cents_only = (
            pynutil.delete('integer_part: "0"')
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('fractional_part: "')
            + pynini.closure(NEMO_DIGIT, 1, 3)
            + pynutil.delete('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.delete('currency: "')
            + cents
            + pynutil.delete('"')
        )

        fractional = (pynutil.insert("0") + NEMO_DIGIT) | pynini.closure(
            NEMO_DIGIT, 1, 2
        )
        graph_fractional_only = (
            pynutil.delete('integer_part: "')
            + pynini.accep("0")
            + pynutil.delete('"')
            + pynini.cross(NEMO_SPACE, ",")
            + pynutil.delete('fractional_part: "')
            + fractional
            + pynutil.delete('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.delete('currency: "')
            + currency_symbol
            + pynutil.delete('"')
        )

        graph_major_fractional = (
            pynutil.delete('integer_part: "')
            + (
                pynini.closure(integer_components)
                @ pynini.cdrewrite(get_powers_of_ten, "", "[EOS]", NEMO_SIGMA)
                @ pynini.cdrewrite(
                    pynutil.insert(NEMO_SPACE),
                    NEMO_DIGIT,
                    pynini.project(get_powers_of_ten, "output"),
                    NEMO_SIGMA,
                )
            )
            + pynutil.delete('"')
            + pynini.cross(NEMO_SPACE, ",")
            + pynutil.delete('fractional_part: "')
            + fractional
            + pynutil.delete('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.delete('currency: "')
            + currency_symbol
            + pynutil.delete('"')
        )

        graph_money_final = (
            graph_major_only
            | graph_cents_only
            | graph_fractional_only
            | graph_major_fractional
        )

        remove_leading_zero_clusters = pynini.cdrewrite(
            pynutil.delete("000 "),
            "[BOS]",
            "",
            NEMO_SIGMA,
        )

        graph_money_final = graph_money_final @ remove_leading_zero_clusters

        delete_tokens = self.delete_tokens(graph_money_final)
        self.fst = delete_tokens.optimize()
