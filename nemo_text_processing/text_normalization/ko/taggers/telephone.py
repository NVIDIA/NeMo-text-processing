# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import (
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    한국어 전화번호 태거(미니멀)
    - 0은 항상 "영"
    - 패턴: [ +<cc> ] (### | (###)) ( - | . ) (### | ####) ( - | . ) ####
    - 확장/SSN/IP 미지원
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        add_sep = pynutil.insert(", ")  # standard block separator ", "

        # 1) safe digit mapping: force 0 -> "영" (do not rely on zero.tsv invert)
        digit = pynini.string_file(get_abs_path("data/number/digit.tsv")).optimize()
        zero_map = pynini.cross("0", "영")
        digit_ko = (digit | zero_map).optimize()

        # helper: n digits as per-digit reading ("일 이 삼" ...)
        def n_digits(n: int):
            return (digit_ko + pynini.closure(digit_ko, 0, n - 1)).optimize()

        def three_digits():
            return n_digits(3)

        def four_digits():
            return n_digits(4)

        # country code: "+1", "+82", "+1-"
        country_core = (
            pynini.cross("+", "플러스 ")
            + pynini.closure(digit_ko + insert_space, 0, 2)
            + digit_ko
            + pynutil.insert(",")
        )
        country_code = pynutil.insert('country_code: "') + country_core + pynutil.insert('"')
        country_code = country_code + pynini.closure(pynutil.delete("-"), 0, 1) + delete_space + insert_space

        # area part: "123-" | "123." | "(123)" [space?] or "(123)-"
        area_core = three_digits()
        area_part = (
            (area_core + (pynutil.delete("-") | pynutil.delete(".")))
            | (
                pynutil.delete("(")
                + area_core
                + ((pynutil.delete(")") + pynini.closure(pynutil.delete(" "), 0, 1)) | pynutil.delete(")-"))
            )
        ) + add_sep

        # 2) allow 3 **or 4** digits in the middle block (to support 010-3713-7050)
        mid = pynini.union(three_digits(), four_digits()).optimize()
        last4 = four_digits()

        # consume '-' or '.' between middle and last blocks
        number_part_core = area_part + mid + (pynutil.delete("-") | pynutil.delete(".")) + add_sep + last4
        number_part = pynutil.insert('number_part: "') + number_part_core + pynutil.insert('"')

        # final graph: with or without country code
        graph = pynini.union(country_code + number_part, number_part).optimize()

        self.fst = self.add_tokens(graph).optimize()
