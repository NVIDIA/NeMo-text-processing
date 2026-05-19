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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { currency_maj: "रुपए" integer_part: "बारह" } } -> बारह रुपए
        money { currency_maj: "रुपए" integer_part: "बारह" fractional_part: "पचास" currency_min: "पैसे" } -> बारह रुपए पचास पैसे
        money { currency_maj: "रुपए" integer_part: "शून्य" fractional_part: "पचास" currency_min: "पैसे" } -> पचास पैसे

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        sp = pynini.accep(NEMO_SPACE)

        currency_major = pynutil.delete('currency_maj: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        fractional_part = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        currency_minor = pynutil.delete('currency_min: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        graph_major_only = integer_part + sp + currency_major

        all_major_names = [maj for maj, _ in load_labels(get_abs_path("data/money/major_minor_currencies.tsv"))]

        major_minor_graphs = []
        minor_only_graphs = []

        for major in all_major_names:
            graph_major_slot = pynutil.delete('currency_maj: "') + pynutil.delete(major) + pynutil.delete('"')

            major_minor_graphs.append(
                graph_major_slot
                + sp
                + integer_part
                + pynutil.insert(NEMO_SPACE)
                + pynutil.insert(major)
                + sp
                + fractional_part
                + sp
                + currency_minor
            )

            minor_only_graphs.append(
                graph_major_slot
                + sp
                + pynutil.delete('integer_part: "शून्य"')
                + sp
                + fractional_part
                + sp
                + currency_minor
            )

        graph_major_minor = pynini.union(*major_minor_graphs)
        graph_minor_only = pynini.union(*minor_only_graphs)

        decimal_graphs = []
        for major in all_major_names:
            decimal_graphs.append(
                pynutil.delete('currency_maj: "')
                + pynutil.delete(major)
                + pynutil.delete('"')
                + sp
                + integer_part
                + sp
                + pynutil.insert(" दशमलव ")
                + fractional_part
                + pynutil.insert(NEMO_SPACE)
                + pynutil.insert(major)
            )
        graph_decimal_money = pynini.union(*decimal_graphs)

        graph = graph_major_only | graph_major_minor | pynutil.add_weight(graph_minor_only, -0.1) | graph_decimal_money

        self.fst = self.delete_tokens(graph).optimize()
