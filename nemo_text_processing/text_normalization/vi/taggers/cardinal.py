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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class CardinalFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        resources = {
            'zero': pynini.string_file(get_abs_path("data/numbers/zero.tsv")),
            'digit': pynini.string_file(get_abs_path("data/numbers/digit.tsv")),
            'teen': pynini.string_file(get_abs_path("data/numbers/teen.tsv")),
            'ties': pynini.string_file(get_abs_path("data/numbers/ties.tsv")),
        }
        self.zero, self.digit, self.teen, self.ties = resources.values()

        with open(get_abs_path("data/numbers/magnitudes.tsv"), 'r', encoding='utf-8') as f:
            self.magnitudes = {parts[0]: parts[1] for line in f if len(parts := line.strip().split('\t')) == 2}

        with open(get_abs_path("data/numbers/digit_special.tsv"), 'r', encoding='utf-8') as f:
            special = {
                parts[0]: {'std': parts[1], 'alt': parts[2]}
                for line in f
                if len(parts := line.strip().split('\t')) >= 3
            }

        self.special_digits = pynini.union(
            *[pynini.cross(k, v["alt"]) for k, v in special.items() if k in ["1", "4", "5"]]
        )
        self.linh_digits = pynini.union(*[pynini.cross(k, special[k]["std"]) for k in ["1", "4", "5"]], self.digit)

        self.single_digit = self.digit

        self.two_digit = pynini.union(
            self.teen,
            self.ties + pynutil.delete("0"),
            self.ties
            + insert_space
            + pynini.union(self.special_digits, pynini.union("2", "3", "6", "7", "8", "9") @ self.digit),
        )

        self.hundreds_pattern = pynini.union(
            self.single_digit + insert_space + pynutil.insert(self.magnitudes["hundred"]) + pynutil.delete("00"),
            self.single_digit
            + insert_space
            + pynutil.insert(self.magnitudes["hundred"])
            + pynutil.delete("0")
            + insert_space
            + pynutil.insert(self.magnitudes["linh"])
            + insert_space
            + self.linh_digits,
            self.single_digit
            + insert_space
            + pynutil.insert(self.magnitudes["hundred"])
            + insert_space
            + self.two_digit,
        )

        self.hundreds = pynini.closure(NEMO_DIGIT, 3, 3) @ self.hundreds_pattern

        self.thousand = self._build_magnitude_pattern("thousand", 4, 6, 3)
        self.million = self._build_magnitude_pattern("million", 7, 9, 6, self.thousand)
        self.billion = self._build_magnitude_pattern("billion", 10, 12, 9, self.million)

        self.graph = pynini.union(
            self.billion, self.million, self.thousand, self.hundreds, self.two_digit, self.single_digit, self.zero
        ).optimize()

        self.single_digits_graph = self.single_digit | self.zero
        self.graph_with_and = self.graph

        self.fst = self.add_tokens(
            pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
            + pynutil.insert("integer: \"")
            + self.graph
            + pynutil.insert("\"")
        ).optimize()

    def _build_magnitude_pattern(self, name, min_digits, max_digits, zero_count, prev_pattern=None):
        magnitude_word = self.magnitudes[name]

        patterns = []
        for digits in range(min_digits, max_digits + 1):
            leading_digits = digits - zero_count
            leading_fst = {1: self.single_digit, 2: self.two_digit, 3: self.hundreds_pattern}.get(
                leading_digits, self.hundreds_pattern
            )

            prefix = leading_fst + insert_space + pynutil.insert(magnitude_word)

            digit_patterns = [prefix + pynutil.delete("0" * zero_count)]

            if prev_pattern:
                digit_patterns.append(prefix + insert_space + prev_pattern)

            trailing_patterns = []
            for trailing_zeros in range(zero_count):
                remaining_digits = zero_count - trailing_zeros
                if remaining_digits == 1:
                    trailing_patterns.append(
                        prefix
                        + pynutil.delete("0" * trailing_zeros)
                        + insert_space
                        + pynutil.insert(self.magnitudes["linh"])
                        + insert_space
                        + self.linh_digits
                    )
                elif remaining_digits == 2:
                    trailing_patterns.append(
                        prefix + pynutil.delete("0" * trailing_zeros) + insert_space + self.two_digit
                    )
                elif remaining_digits == 3:
                    trailing_patterns.append(
                        prefix + pynutil.delete("0" * trailing_zeros) + insert_space + self.hundreds_pattern
                    )
            digit_patterns.extend(trailing_patterns)

            if name == "million" and digits == 7:
                digit_patterns.extend(
                    [
                        prefix
                        + pynutil.delete("00")
                        + insert_space
                        + self.single_digit
                        + insert_space
                        + pynutil.insert(self.magnitudes["thousand"])
                        + pynutil.delete("00")
                        + insert_space
                        + pynutil.insert(self.magnitudes["linh"])
                        + insert_space
                        + self.linh_digits,
                        prefix
                        + pynutil.delete("0")
                        + insert_space
                        + self.two_digit
                        + insert_space
                        + pynutil.insert(self.magnitudes["thousand"])
                        + pynutil.delete("00")
                        + insert_space
                        + pynutil.insert(self.magnitudes["linh"])
                        + insert_space
                        + self.linh_digits,
                    ]
                )
            elif name == "billion" and digits == 10:
                digit_patterns.append(
                    prefix
                    + pynutil.delete("00")
                    + insert_space
                    + self.single_digit
                    + insert_space
                    + pynutil.insert(self.magnitudes["million"])
                    + pynutil.delete("00")
                    + insert_space
                    + self.single_digit
                    + insert_space
                    + pynutil.insert(self.magnitudes["thousand"])
                    + insert_space
                    + self.hundreds_pattern
                )

            patterns.append(pynini.closure(NEMO_DIGIT, digits, digits) @ pynini.union(*digit_patterns))

        return pynini.union(*patterns)
