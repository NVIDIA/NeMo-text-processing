# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class CardinalFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)
        # Load base .tsv files
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))

        digit_except_one = pynini.difference(NEMO_DIGIT, "1")
        digit_except_zero_one = pynini.difference(digit_except_one, "0")

        graph_digit_no_zero_one = digit_except_zero_one @ graph_digit
        graph_ty = pynini.string_file(get_abs_path("data/number/ty.tsv"))

        # Compose all basic number forms
        graph_1_to_99 = (graph_ty + (graph_digit | pynutil.delete('0'))) | graph_digit

        hundreds = NEMO_DIGIT**3
        graph_hundred_component = (
            pynini.cross('1', '백') | (graph_digit_no_zero_one + pynutil.insert('백'))
        ) + pynini.union(pynini.closure(pynutil.delete('0')), (pynini.closure(pynutil.delete('0')) + graph_1_to_99))
        graph_hundred = hundreds @ graph_hundred_component

        thousands = NEMO_DIGIT**4
        graph_thousand_component = (
            pynini.cross('1', '천') | (graph_digit_no_zero_one + pynutil.insert('천'))
        ) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_hundred_component,
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_thousand = thousands @ graph_thousand_component

        ten_thousands = NEMO_DIGIT**5
        graph_ten_thousand_component = (pynini.cross('1', '만') | (graph_digit + pynutil.insert('만'))) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_ten_thousand = ten_thousands @ graph_ten_thousand_component

        hundred_thousands = NEMO_DIGIT**6

        graph_hundred_thousand_component = ((NEMO_DIGIT**2 @ graph_1_to_99) + pynutil.insert('만')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_hundred_thousand = hundred_thousands @ graph_hundred_thousand_component

        millions = NEMO_DIGIT**7
        graph_million_component = ((graph_hundred) + pynutil.insert('만')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_million = millions @ graph_million_component

        ten_millions = NEMO_DIGIT**8
        graph_ten_million_component = ((graph_thousand) + pynutil.insert('만')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_ten_million = ten_millions @ graph_ten_million_component

        hundred_millions = NEMO_DIGIT**9
        graph_hundred_million_component = (graph_digit + pynutil.insert('억')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + graph_million_component),
            (pynutil.delete('00') + graph_hundred_thousand_component),
            (pynutil.delete('000') + graph_ten_thousand_component),
            (pynutil.delete('0000') + graph_thousand_component),
            ((pynutil.delete('00000') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_hundred_million = hundred_millions @ graph_hundred_million_component

        thousand_millions = NEMO_DIGIT**10
        graph_thousand_million_component = ((NEMO_DIGIT**2 @ graph_1_to_99) + pynutil.insert('억')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + graph_million_component),
            (pynutil.delete('00') + graph_hundred_thousand_component),
            (pynutil.delete('000') + graph_ten_thousand_component),
            (pynutil.delete('0000') + graph_thousand_component),
            ((pynutil.delete('00000') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_thousand_million = thousand_millions @ graph_thousand_million_component

        billions = NEMO_DIGIT**11
        graph_billions_component = ((graph_hundred) + pynutil.insert('억')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + graph_million_component),
            (pynutil.delete('00') + graph_hundred_thousand_component),
            (pynutil.delete('000') + graph_ten_thousand_component),
            (pynutil.delete('0000') + graph_thousand_component),
            ((pynutil.delete('00000') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_billions = billions @ graph_billions_component

        ten_billions = NEMO_DIGIT**12
        graph_ten_billions_component = ((graph_thousand) + pynutil.insert('억')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + graph_million_component),
            (pynutil.delete('00') + graph_hundred_thousand_component),
            (pynutil.delete('000') + graph_ten_thousand_component),
            (pynutil.delete('0000') + graph_thousand_component),
            ((pynutil.delete('00000') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_ten_billions = ten_billions @ graph_ten_billions_component

        hundred_billions = NEMO_DIGIT**13
        graph_hundred_billions_component = (graph_digit + pynutil.insert('조')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_billions_component,
            pynutil.delete('0') + graph_billions_component,
            pynutil.delete('00') + graph_thousand_million_component,
            pynutil.delete('000') + graph_hundred_million_component,
            pynutil.delete('0000') + graph_ten_million_component,
            pynutil.delete('00000') + graph_million_component,
            pynutil.delete('000000') + graph_hundred_thousand_component,
            pynutil.delete('0000000') + graph_ten_thousand_component,
            pynutil.delete('00000000') + graph_thousand_component,
            pynutil.delete('000000000') + graph_hundred_component,
            (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
        )
        graph_hundred_billions = hundred_billions @ graph_hundred_billions_component

        trillion = NEMO_DIGIT**14
        graph_trillion_component = (
            (NEMO_DIGIT**2 @ graph_1_to_99)
            + pynutil.insert('조')
            + pynini.union(
                pynini.closure(pynutil.delete('0')),
                graph_ten_billions_component,
                pynutil.delete('0') + graph_billions_component,
                pynutil.delete('00') + graph_thousand_million_component,
                pynutil.delete('000') + graph_hundred_million_component,
                pynutil.delete('0000') + graph_ten_million_component,
                pynutil.delete('00000') + graph_million_component,
                pynutil.delete('000000') + graph_hundred_thousand_component,
                pynutil.delete('0000000') + graph_ten_thousand_component,
                pynutil.delete('00000000') + graph_thousand_component,
                pynutil.delete('000000000') + graph_hundred_component,
                (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
            )
        )
        graph_trillions = trillion @ graph_trillion_component

        ten_trillions = NEMO_DIGIT**15
        graph_ten_trillions_component = (
            (graph_hundred)
            + pynutil.insert('조')
            + pynini.union(
                pynini.closure(pynutil.delete('0')),
                graph_ten_billions_component,
                pynutil.delete('0') + graph_billions_component,
                pynutil.delete('00') + graph_thousand_million_component,
                pynutil.delete('000') + graph_hundred_million_component,
                pynutil.delete('0000') + graph_ten_million_component,
                pynutil.delete('00000') + graph_million_component,
                pynutil.delete('000000') + graph_hundred_thousand_component,
                pynutil.delete('0000000') + graph_ten_thousand_component,
                pynutil.delete('00000000') + graph_thousand_component,
                pynutil.delete('000000000') + graph_hundred_component,
                (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
            )
        )
        graph_ten_trillions = ten_trillions @ graph_ten_trillions_component

        hundred_trillions = NEMO_DIGIT**16
        graph_hundred_trillions_component = (
            (graph_thousand)
            + pynutil.insert('조')
            + pynini.union(
                pynini.closure(pynutil.delete('0')),
                graph_ten_billions_component,
                pynutil.delete('0') + graph_billions_component,
                pynutil.delete('00') + graph_thousand_million_component,
                pynutil.delete('000') + graph_hundred_million_component,
                pynutil.delete('0000') + graph_ten_million_component,
                pynutil.delete('00000') + graph_million_component,
                pynutil.delete('000000') + graph_hundred_thousand_component,
                pynutil.delete('0000000') + graph_ten_thousand_component,
                pynutil.delete('00000000') + graph_thousand_component,
                pynutil.delete('000000000') + graph_hundred_component,
                (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
            )
        )
        graph_hundred_trillions = hundred_trillions @ graph_hundred_trillions_component

        thousand_trillions = NEMO_DIGIT**17
        graph_thousand_trillions_component = (
            graph_digit
            + pynutil.insert('경')
            + pynini.union(
                pynini.closure(pynutil.delete('0')),
                graph_hundred_trillions_component,
                pynutil.delete('0') + graph_ten_trillions_component,
                pynutil.delete('00') + graph_trillion_component,
                pynutil.delete('000') + graph_hundred_billions_component,
                pynutil.delete('0000') + graph_ten_billions_component,
                pynutil.delete('00000') + graph_billions_component,
                pynutil.delete('000000') + graph_thousand_million_component,
                pynutil.delete('0000000') + graph_hundred_million_component,
                pynutil.delete('00000000') + graph_ten_million_component,
                pynutil.delete('000000000') + graph_million_component,
                pynutil.delete('0000000000') + graph_hundred_thousand_component,
                pynutil.delete('00000000000') + graph_ten_thousand_component,
                pynutil.delete('000000000000') + graph_thousand_component,
                pynutil.delete('0000000000000') + graph_hundred_component,
                (pynini.closure(pynutil.delete('0')) + graph_1_to_99),
            )
        )
        graph_thousand_trillions = thousand_trillions @ graph_thousand_trillions_component

        # FST
        graph_num = pynini.union(
            graph_thousand_trillions,
            graph_hundred_trillions,
            graph_ten_trillions,
            graph_trillions,
            graph_hundred_billions,
            graph_ten_billions,
            graph_billions,
            graph_thousand_million,
            graph_hundred_million,
            graph_ten_million,
            graph_million,
            graph_hundred_thousand,
            graph_ten_thousand,
            graph_thousand,
            graph_hundred,
            graph_1_to_99,
            graph_zero,
        ).optimize()

        # Sign and final formatting
        optional_sign = pynini.closure(pynutil.insert('negative: "true" ') + pynini.cross("-", ""), 0, 1)
        final_graph = optional_sign + pynutil.insert('integer: "') + graph_num + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
