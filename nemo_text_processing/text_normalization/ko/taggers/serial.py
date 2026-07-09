# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying Korean serial/number-like strings.

    This class is signal-based, similar to MoneyFst. It only reads digits
    one by one when the input contains a clear signal such as "번호" or "연락처".

    Example inputs and outputs:
        번호는 0987654321 -> name: "번호는 영구팔칠육오사삼이일"
        휴대폰 번호는 0987654321 -> tokens { name: "휴대폰" } tokens { name: "번호는 영구팔칠육오사삼이일" }
        연락처는 12345678 -> name: "연락처는 일이삼사오육칠팔"

    Args:
        deterministic: If True, provide a single transduction;
            if False, allow multiple transductions.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        sp = pynini.closure(delete_space)

        # Digit mapping. Force 0 -> "영" for serial/number readings.
        digit = pynini.string_file(get_abs_path("data/number/digit.tsv")).optimize()
        zero_map = pynini.cross("0", "영")
        digit_ko = (digit | zero_map).optimize()

        # Optional separators inside number-like strings.
        # These separators are deleted so that the number is read digit-by-digit.
        # Examples:
        #   123-456 -> 일이삼사오육
        #   123.456 -> 일이삼사오육
        #   123 456 -> 일이삼사오육
        sep = pynutil.delete("-") | pynutil.delete(".") | pynutil.delete(" ")

        # Require at least 3 digits.
        # This covers common serial-like numbers such as reservation numbers,
        # verification numbers, account numbers, and context-based phone-number strings.
        # Very short numbers such as "번호는 12" are left to the existing cardinal path.
        min_three_digits = (
            digit_ko
            + pynini.closure(sep, 0, 1)
            + digit_ko
            + pynini.closure(sep, 0, 1)
            + digit_ko
        )

        serial_body = (
            min_three_digits
            + pynini.closure(pynini.closure(sep, 0, 1) + digit_ko)
        ).optimize()

        # Minimal context signals.
        # "번호는" covers 휴대폰 번호는, 전화 번호는, 계좌 번호는, 예약 번호는, etc.,
        # because the preceding noun can stay outside the serial token.
        signal = pynini.string_map([
            ("번호는", "번호는"),
            ("번호가", "번호가"),
            ("번호를", "번호를"),

            ("연락처는", "연락처는"),
            ("연락처가", "연락처가"),
            ("연락처를", "연락처를"),
        ]).optimize()

        graph = (
            pynutil.insert('name: "')
            + signal
            + pynutil.insert('" } tokens { name: "')
            + sp
            + serial_body
            + pynutil.insert('"')
        ).optimize()

        self.fst = graph.optimize()