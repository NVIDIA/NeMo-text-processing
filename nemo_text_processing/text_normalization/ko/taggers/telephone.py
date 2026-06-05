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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Korean telephone numbers.

    Example inputs → tokens:
        +82 010-3713-7050  -> telephone { country_code: "국가번호 팔이," number_part: "영일영 삼칠일삼 칠영오영" }
        +1 (415) 555-0123 -> telephone { country_code: "국가번호 일,"   number_part: "사일오 오오오 영일이삼" }
        (031)371-3700     -> telephone { number_part: "영삼일 삼칠일 삼칠영영" }
        010-3713-7050     -> telephone { number_part: "영일영 삼칠일삼 칠영오영" }
        010.777.8888      -> telephone { number_part: "영일영 칠칠칠 팔팔팔팔" }

    Args:
        deterministic (bool, optional): If True, provide a single transduction;
            if False, allow multiple transductions.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)
        # Separator between digit blocks (e.g., "-" or ".")
        delete_sep = pynutil.delete("-") | pynutil.delete(".")
        # Optional space inserted between blocks
        insert_block_space = insert_space

        # 1) safe digit mapping: force 0 -> "영" (do not rely on zero.tsv invert)
        digit = pynini.string_file(get_abs_path("data/number/digit.tsv")).optimize()
        zero_map = pynini.cross("0", "영")
        digit_ko = (digit | zero_map).optimize()

        three_digits = digit_ko**3
        four_digits = digit_ko**4

        # country code: "+1", "+82", "+1-"
        cc_digits = pynini.closure(digit_ko, 1, 3)

        country_code = (
            pynutil.delete("+")
            + pynutil.insert('country_code: "')
            + cc_digits
            + pynutil.insert('"')
            + pynini.closure(pynutil.delete("-") | pynutil.delete(" "), 0, 1)
            + delete_space
        )

        # area part: "123-" | "123." | "(123)" [space?] or "(123)-"
        area_core = three_digits
        area_part = (
            (area_core + delete_sep)
            | (
                pynutil.delete("(")
                + area_core
                + pynutil.delete(")")
                + pynini.closure(pynutil.delete(" "), 0, 1)
                + pynini.closure(delete_sep, 0, 1)
            )
        ) + insert_block_space

        # 2) allow 3 **or 4** digits in the middle block (to support 010-3713-7050)
        mid = pynini.union(three_digits, four_digits)
        last4 = four_digits

        # consume '-' or '.' between middle and last blocks
        number_part_core = area_part + mid + delete_sep + insert_block_space + last4
        number_part = pynutil.insert('number_part: "') + number_part_core + pynutil.insert('"')

        # final graph: with or without country code
        graph = pynini.union(country_code + insert_space + number_part, number_part).optimize()

        self.fst = self.add_tokens(graph).optimize()
