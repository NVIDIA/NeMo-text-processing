# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying Japanese telephone numbers.

    Examples:
        090-1234-5678 -> telephone { number_part: "ゼロ九ゼロ 一二三四 五六七八" preserve_order: true }
        03-1234-5678 -> telephone { number_part: "ゼロ三 一二三四 五六七八" preserve_order: true }
        +81 90-1234-5678 -> telephone { country_code: "八一" number_part: "九ゼロ 一二三四 五六七八" preserve_order: true }

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        digit = graph_digit | graph_zero
        extension_cue = pynini.project(
            pynini.string_file(get_abs_path("data/telephone/extension.tsv")),
            "input",
        )

        sep_char = pynini.union("-", "－", "ー", ".", "．")
        delete_sep = pynutil.delete(sep_char)
        delete_required_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE, 1))
        block_sep = delete_space + (delete_sep | delete_required_space) + delete_space + insert_space
        optional_after_paren_sep = delete_space + pynini.closure(delete_sep + delete_space, 0, 1)
        required_country_sep = delete_space + (delete_sep | delete_required_space) + delete_space

        open_paren = pynutil.delete("(") | pynutil.delete("（")
        close_paren = pynutil.delete(")") | pynutil.delete("）")

        digits = {count: digit**count for count in range(1, 11)}
        paren = {
            count: open_paren + digits[count] + close_paren + optional_after_paren_sep + insert_space
            for count in range(1, 4)
        }

        local_grouped_number = (
            digits[2] + block_sep + digits[4] + block_sep + digits[4]
            | digits[3] + block_sep + digits[3] + block_sep + digits[4]
            | digits[3] + block_sep + digits[4] + block_sep + digits[4]
            | digits[4] + block_sep + digits[2] + block_sep + digits[4]
            | digits[4] + block_sep + digits[3] + block_sep + digits[3]
            | digits[4] + block_sep + digits[3] + block_sep + digits[4]
            | digits[4] + block_sep + digits[4] + block_sep + digits[3]
        )

        local_parenthesized_number = (
            paren[2] + digits[4] + block_sep + digits[4]
            | paren[3] + digits[3] + block_sep + digits[4]
            | paren[3] + digits[4] + block_sep + digits[4]
        )

        compact_local_number = graph_zero + (digits[9] | digits[10])
        local_number = local_grouped_number | local_parenthesized_number | compact_local_number

        international_number = (
            digits[1] + block_sep + digits[4] + block_sep + digits[4]
            | digits[2] + block_sep + digits[3] + block_sep + digits[4]
            | digits[2] + block_sep + digits[4] + block_sep + digits[4]
            | digits[3] + block_sep + digits[2] + block_sep + digits[4]
            | digits[3] + block_sep + digits[3] + block_sep + digits[4]
            | paren[1] + digits[4] + block_sep + digits[4]
            | paren[2] + digits[3] + block_sep + digits[4]
            | paren[2] + digits[4] + block_sep + digits[4]
            | paren[3] + digits[2] + block_sep + digits[4]
            | paren[3] + digits[3] + block_sep + digits[4]
        )

        country_code = digits[1] | digits[2]
        country_code_component = (
            (pynutil.delete("+") | pynutil.delete("＋"))
            + pynutil.insert('country_code: "')
            + country_code
            + pynutil.insert('"')
            + required_country_sep
            + pynutil.insert(" ")
        )

        extension = (
            delete_space
            + pynutil.delete(extension_cue)
            + delete_space
            + pynutil.insert(' extension: "')
            + pynini.closure(digit, 1, 4)
            + pynutil.insert('"')
        )

        number_part = pynutil.insert('number_part: "') + local_number + pynutil.insert('"')
        international_number_part = pynutil.insert('number_part: "') + international_number + pynutil.insert('"')

        graph = number_part | (country_code_component + international_number_part)
        graph = graph + pynini.closure(extension, 0, 1)
        graph = graph + pynutil.insert(" preserve_order: true")

        self.fst = self.add_tokens(graph.optimize()).optimize()
