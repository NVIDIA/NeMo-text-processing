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

        # Load base digits
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_hundred = pynini.string_file(get_abs_path("data/number/hundred.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/number/teen.tsv"))
        graph_thousand = pynini.string_file(get_abs_path("data/number/thousand.tsv"))
        graph_tenthousand = pynini.string_file(get_abs_path("data/number/tenthousand.tsv"))
        graph_ty = pynini.string_file(get_abs_path("data/number/ty.tsv"))

        digit_except_one = pynini.difference(NEMO_DIGIT, "1")
        digit_except_zero_one = pynini.difference(digit_except_one, "0")

        # Custom basic units
        graph_1_to_9 = graph_digit
        graph_10_to_19 = graph_teen
        graph_20_to_99 = graph_ty

        graph_all = pynini.union(
            graph_1_to_9,
            graph_10_to_19,
            graph_20_to_99,
            graph_zero,
        )

        # 1-9 reading
        read_1 = graph_digit

        # 10-19 reading
        read_10_to_19 = graph_teen

        # 20-99 reading
        read_20_to_99 = graph_ty

        hundreds = NEMO_DIGIT**3
        graph_hundred_component = (
            pynini.cross("1", "백") | (digit_except_zero_one @ graph_digit + pynutil.insert("백"))
        ) + pynini.union(pynini.cross("00", ""), pynutil.delete("0") + graph_1_to_9, read_10_to_19, read_20_to_99)
        graph_hundred = hundreds @ graph_hundred_component

        read_100_to_999 = (NEMO_DIGIT**3) @ graph_hundred_component

        thousands = NEMO_DIGIT**4
        graph_thousand_component = (
            pynini.cross("1", "천") | (digit_except_zero_one @ graph_digit + pynutil.insert("천"))
        ) + pynini.union(
            pynini.cross("000", ""),
            pynutil.delete("00") + graph_1_to_9,
            pynutil.delete("0") + read_10_to_19,
            pynutil.delete("0") + read_20_to_99,
            read_100_to_999,
        )
        graph_thousand = thousands @ graph_thousand_component

        # 1-99 reading
        read_1_to_99 = pynini.union(read_1, read_10_to_19, read_20_to_99).optimize()
        read_100_to_999 = (NEMO_DIGIT**3) @ graph_hundred_component
        read_1000_to_9999 = (NEMO_DIGIT**4) @ graph_thousand_component
        read_1_to_999_no_leading_zeros = pynini.union(read_100_to_999, read_1_to_99).optimize()
        read_1_to_9999_no_leading_zeros = pynini.union(read_1000_to_9999, read_100_to_999, read_1_to_99).optimize()

        # 1~9999 (No 0)
        read_1_to_4_digits_improved = pynini.union(
            pynini.cross("0000", ""),
            pynini.cross("000", "") + read_1,
            pynini.cross("00", "") + read_1_to_99,
            pynini.cross("0", "") + read_1_to_999_no_leading_zeros,
            read_1_to_9999_no_leading_zeros,
        ).optimize()
        read_optional_1_to_4_digits = read_1_to_4_digits_improved

        ten_thousands = NEMO_DIGIT**5
        graph_tenthousand_component = (
            pynini.cross("1", "만") | (digit_except_zero_one @ graph_digit + pynutil.insert("만"))
        ) + pynini.union(
            pynini.closure(pynutil.delete("0")),
            graph_thousand_component,
            (pynini.closure(pynutil.delete("0")) + graph_hundred_component),
            (pynini.closure(pynutil.delete("0")) + graph_all),
        )
        graph_tenthousand = ten_thousands @ graph_tenthousand_component

        hundred_thousands = NEMO_DIGIT**6
        graph_hundredthousand_component = (
            (NEMO_DIGIT**2 @ read_1_to_99) + pynutil.insert("만 ") + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
        )
        graph_hundredthousand = hundred_thousands @ graph_hundredthousand_component

        millions = NEMO_DIGIT**7
        graph_million_component = (
            (NEMO_DIGIT**3 @ read_100_to_999) + pynutil.insert("만 ") + (NEMO_DIGIT**4 @ read_1_to_4_digits_improved)
        )
        graph_million = millions @ graph_million_component

        ten_millions = NEMO_DIGIT**8
        first_part = (NEMO_DIGIT**4 @ read_1000_to_9999) + pynutil.insert("만 ")

        graph_tenmillion_component = pynini.union(pynini.cross("0000", ""), first_part) + (
            NEMO_DIGIT**4 @ read_optional_1_to_4_digits
        )
        graph_tenmillion = ten_millions @ graph_tenmillion_component

        hundred_millions = NEMO_DIGIT**9
        read_8_digits = pynini.union(
            pynini.cross("00000000", ""),
            pynini.cross("0000", "") + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
            (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
            + pynutil.insert("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
        ).optimize()

        graph_hundredmillion_component = (NEMO_DIGIT**1 @ read_1) + pynutil.insert("억 ") + read_8_digits
        graph_hundredmillion = hundred_millions @ graph_hundredmillion_component

        billions = NEMO_DIGIT**10
        read_8_digits_for_billion = pynini.union(
            pynini.cross("00000000", ""),
            pynini.cross("0000", "") + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
            (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
            + pynutil.insert("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
        ).optimize()

        graph_billion_component = (NEMO_DIGIT**2 @ read_1_to_99) + pynutil.insert("억 ") + read_8_digits_for_billion
        graph_billion = billions @ graph_billion_component

        ten_billions = NEMO_DIGIT**11
        read_8_digits_for_tenbillion = pynini.union(
            pynini.cross("00000000", ""),
            pynini.cross("0000", "") + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
            (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
            + pynutil.insert("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
        ).optimize()

        graph_tenbillion_component = (
            (NEMO_DIGIT**3 @ read_100_to_999) + pynutil.insert("억 ") + read_8_digits_for_tenbillion
        )
        graph_ten_billion = ten_billions @ graph_tenbillion_component

        hundred_billions = NEMO_DIGIT**12
        read_8_digits_for_hundredbillion = pynini.union(
            pynini.cross("00000000", ""),
            pynini.cross("0000", "") + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
            (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
            + pynutil.insert("만")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits),
        ).optimize()

        graph_hundredbillion_component = (
            (NEMO_DIGIT**4 @ read_1000_to_9999) + pynutil.insert("억 ") + read_8_digits_for_hundredbillion
        )
        graph_hundred_billion = hundred_billions @ graph_hundredbillion_component

        trillions = NEMO_DIGIT**13

        def read_4_optional_with_unit(unit: str):
            return pynini.union(
                pynini.cross("0000", ""), (NEMO_DIGIT**4 @ read_optional_1_to_4_digits) + pynutil.insert(unit)
            )

        graph_trillion_component = (
            (NEMO_DIGIT**1 @ read_1)
            + pynutil.insert("조 ")
            + read_4_optional_with_unit("억 ")
            + read_4_optional_with_unit("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
        )
        graph_trillion = trillions @ graph_trillion_component

        ten_trillions = NEMO_DIGIT**14

        def read_4_optional_with_unit(unit: str):
            return pynini.union(
                pynini.cross("0000", ""), (NEMO_DIGIT**4 @ read_optional_1_to_4_digits) + pynutil.insert(unit)
            )

        graph_ten_trillion_component = (
            (NEMO_DIGIT**2 @ read_1_to_99)
            + pynutil.insert("조 ")
            + read_4_optional_with_unit("억 ")
            + read_4_optional_with_unit("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
        )
        graph_ten_trillion = ten_trillions @ graph_ten_trillion_component

        hundred_trillions = NEMO_DIGIT**15

        def read_4_optional_with_unit(unit: str):
            return pynini.union(
                pynini.cross("0000", ""), (NEMO_DIGIT**4 @ read_optional_1_to_4_digits) + pynutil.insert(unit)
            )

        graph_hundred_trillion_component = (
            (NEMO_DIGIT**3 @ read_100_to_999)
            + pynutil.insert("조 ")
            + read_4_optional_with_unit("억 ")
            + read_4_optional_with_unit("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
        )
        graph_hundred_trillion = hundred_trillions @ graph_hundred_trillion_component

        thousand_trillions = NEMO_DIGIT**16

        def read_4_optional_with_unit(unit: str):
            return pynini.union(
                pynini.cross("0000", ""), (NEMO_DIGIT**4 @ read_optional_1_to_4_digits) + pynutil.insert(unit)
            )

        graph_thousand_trillion_component = (
            (NEMO_DIGIT**4 @ read_1000_to_9999)
            + pynutil.insert("조 ")
            + read_4_optional_with_unit("억 ")
            + read_4_optional_with_unit("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
        )
        graph_thousand_trillion = thousand_trillions @ graph_thousand_trillion_component

        quadrillion = NEMO_DIGIT**17

        def read_4_optional_with_unit(unit: str):
            return pynini.union(
                pynini.cross("0000", ""), (NEMO_DIGIT**4 @ read_optional_1_to_4_digits) + pynutil.insert(unit)
            )

        graph_quadrillion_component = (
            (NEMO_DIGIT**1 @ read_1)
            + pynutil.insert("경 ")
            + read_4_optional_with_unit("조 ")
            + read_4_optional_with_unit("억 ")
            + read_4_optional_with_unit("만 ")
            + (NEMO_DIGIT**4 @ read_optional_1_to_4_digits)
        )
        graph_quadrillion = quadrillion @ graph_quadrillion_component

        optional_sign = pynini.closure(pynutil.insert('negative: "true" ') + pynini.cross("-", ""), 0, 1)

        graph_num = pynini.union(
            graph_quadrillion,
            graph_thousand_trillion,
            graph_hundred_trillion,
            graph_ten_trillion,
            graph_trillion,
            graph_hundred_billion,
            graph_ten_billion,
            graph_billion,
            graph_hundredmillion,
            graph_tenmillion,
            graph_million,
            graph_hundredthousand,
            graph_tenthousand,
            graph_thousand,
            graph_hundred,
            graph_20_to_99,
            graph_10_to_19,
            graph_1_to_9,
            read_1,
            graph_zero,
        )

        final_graph = optional_sign + pynutil.insert('integer: "') + graph_num + pynutil.insert('"')

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
