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
        # Separator between number blocks.
        delete_sep = pynutil.delete(pynini.union("-", ".", " ")).optimize()

        # Optional space inserted between blocks
        insert_block_space = insert_space

        # 1) safe digit mapping: force 0 -> "영" (do not rely on zero.tsv invert)
        digit = pynini.string_file(get_abs_path("data/number/digit.tsv")).optimize()
        zero_map = pynini.cross("0", "영")
        digit_ko = (digit | zero_map).optimize()

        two_digits = digit_ko**2
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

        # First block may contain 2 or 3 digits.
        # Examples: 02, 031, 043, 010
        first_block = pynini.union(
            two_digits,
            three_digits,
        ).optimize()

        # Middle block may contain 3 or 4 digits.
        # Examples: 123, 1234
        middle_block = pynini.union(
            three_digits,
            four_digits,
        ).optimize()

        # Plain telephone form:
        #   02-1234-5678
        plain_first_part = (first_block + delete_sep + insert_block_space).optimize()

        # Parenthesized telephone form:
        #   (010)1234-5678
        parenthesized_first_part = (
            pynutil.delete("(")
            + first_block
            + pynutil.delete(")")
            + pynini.closure(
                pynutil.delete(pynini.union(" ", "-", ".")),
                0,
                1,
            )
            + insert_block_space
        ).optimize()

        first_part = pynini.union(
            plain_first_part,
            parenthesized_first_part,
        ).optimize()

        # Standard telephone layout:
        #   2 or 3 digits
        #   followed by 3 or 4 digits
        #   followed by 4 digits
        number_part_core = (first_part + middle_block + delete_sep + insert_block_space + four_digits).optimize()

        number_part = pynutil.insert('number_part: "') + number_part_core + pynutil.insert('"')

        # final graph: with or without country code
        graph = pynini.union(country_code + insert_space + number_part, number_part).optimize()

        self.fst = self.add_tokens(graph).optimize()
