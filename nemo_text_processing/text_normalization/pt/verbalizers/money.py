# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the "License".
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
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
    insert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money (pt-BR), e.g.
        money { currency_maj: "reais" integer_part: "doze" } -> doze reais
        money { ... fractional_part: "cinco" currency_min: "centavos" ... } -> doze reais e cinco centavos

    Args:
        decimal: DecimalFst verbalizer (for decimal amounts embedded in money)
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        scales_data = load_labels(get_abs_path("data/numbers/scales.tsv"))
        currency_plural_data = load_labels(get_abs_path("data/money/currency_major_plural.tsv"))

        scale_words = []
        for row in scales_data[1:]:
            if len(row) < 2:
                continue
            one_label = row[0].strip()
            plural = row[1].strip()
            if not one_label or not plural:
                continue
            scale_words.extend((one_label.split()[-1], plural))

        curr_words = [row[1].strip() for row in currency_plural_data if len(row) >= 2 and row[1].strip()]

        scales = pynini.union(*[pynini.accep(w) + NEMO_SPACE for w in scale_words]).optimize()
        currencies = pynini.union(*curr_words).optimize()

        maj = pynutil.delete('currency_maj: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        min_unit = pynutil.delete('currency_min: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        fractional_part = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )
        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        connector_minor = pynutil.insert("e") + insert_space
        if not deterministic:
            connector_minor |= pynutil.insert("com") + insert_space

        graph_integer = integer_part + NEMO_SPACE + maj

        graph_integer_with_minor = (
            integer_part
            + NEMO_SPACE
            + maj
            + NEMO_SPACE
            + connector_minor
            + fractional_part
            + NEMO_SPACE
            + min_unit
            + delete_preserve_order
        )

        graph_decimal = decimal.numbers + NEMO_SPACE + maj

        graph_minor = fractional_part + NEMO_SPACE + min_unit + delete_preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor
        graph @= pynini.cdrewrite(pynutil.insert("de") + insert_space, scales, currencies, NEMO_SIGMA)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
