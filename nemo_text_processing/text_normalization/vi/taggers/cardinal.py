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

        hundred_word = self.magnitudes["hundred"]
        linh_word = self.magnitudes["linh"]

        self.hundreds_pattern = pynini.union(
            # X00: một trăm, hai trăm, etc.
            self.single_digit + insert_space + pynutil.insert(hundred_word) + pynutil.delete("00"),
            # X0Y: một trăm linh một, hai trăm linh năm, etc.
            self.single_digit
            + insert_space
            + pynutil.insert(hundred_word)
            + pynutil.delete("0")
            + insert_space
            + pynutil.insert(linh_word)
            + insert_space
            + self.linh_digits,
            # XYZ: một trăm hai mười ba, etc.
            self.single_digit + insert_space + pynutil.insert(hundred_word) + insert_space + self.two_digit,
        )

        self.hundreds = pynini.closure(NEMO_DIGIT, 3, 3) @ self.hundreds_pattern

        # Build magnitude patterns (thousands, millions, billions)
        self.thousand = self._build_magnitude_pattern("thousand", 4, 6, 3)
        self.million = self._build_magnitude_pattern("million", 7, 9, 6, self.thousand)
        self.billion = self._build_magnitude_pattern("billion", 10, 12, 9, self.million)

        # Handle dot-separated numbers: 1.000, 1.000.000, etc.
        delete_dot = pynutil.delete(".")
        dot_patterns = []

        # Thousand with dots: 1.000
        dot_patterns.append(
            pynini.compose(
                (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0, 2) + delete_dot + NEMO_DIGIT**3, self.thousand
            )
        )

        # Million with dots: 1.000.000
        dot_patterns.append(
            pynini.compose(
                (NEMO_DIGIT - "0")
                + pynini.closure(NEMO_DIGIT, 0, 2)
                + delete_dot
                + NEMO_DIGIT**3
                + delete_dot
                + NEMO_DIGIT**3,
                self.million,
            )
        )

        # Billion with dots: 1.000.000.000
        dot_patterns.append(
            pynini.compose(
                (NEMO_DIGIT - "0")
                + pynini.closure(NEMO_DIGIT, 0, 2)
                + delete_dot
                + NEMO_DIGIT**3
                + delete_dot
                + NEMO_DIGIT**3
                + delete_dot
                + NEMO_DIGIT**3,
                self.billion,
            )
        )

        self.graph = pynini.union(
            self.billion,
            self.million,
            self.thousand,
            self.hundreds,
            self.two_digit,
            self.single_digit,
            self.zero,
            *dot_patterns,
        ).optimize()

        self.single_digits_graph = self.single_digit | self.zero
        self.graph_with_and = self.graph

        # Build final FST with optional negative and integer wrapper
        negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = negative + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph).optimize()

    def _build_magnitude_pattern(self, name, min_digits, max_digits, zero_count, prev_pattern=None):
        magnitude_word = self.magnitudes[name]
        linh_word = self.magnitudes["linh"]

        patterns = []
        for digits in range(min_digits, max_digits + 1):
            leading_digits = digits - zero_count

            # Choose leading pattern based on digit count
            if leading_digits == 1:
                leading_fst = self.single_digit
            elif leading_digits == 2:
                leading_fst = self.two_digit
            else:  # 3 digits
                leading_fst = self.hundreds_pattern

            prefix = leading_fst + insert_space + pynutil.insert(magnitude_word)
            digit_patterns = []

            # Case 1: All trailing zeros (e.g., 1000 -> một nghìn)
            digit_patterns.append(prefix + pynutil.delete("0" * zero_count))

            # Case 2: Has lower magnitude (e.g., 1001000 -> một triệu một nghìn)
            if prev_pattern:
                digit_patterns.append(prefix + insert_space + prev_pattern)

            # Case 3: Trailing patterns with linh (e.g., 1001 -> một nghìn linh một)
            for trailing_zeros in range(zero_count):
                remaining_digits = zero_count - trailing_zeros
                trailing_prefix = prefix + pynutil.delete("0" * trailing_zeros)

                if remaining_digits == 1:
                    digit_patterns.append(
                        trailing_prefix + insert_space + pynutil.insert(linh_word) + insert_space + self.linh_digits
                    )
                elif remaining_digits == 2:
                    digit_patterns.append(trailing_prefix + insert_space + self.two_digit)
                elif remaining_digits == 3:
                    digit_patterns.append(trailing_prefix + insert_space + self.hundreds_pattern)

            if name == "million" and digits == 7:
                # Handle patterns like 1001001 -> một triệu một nghìn linh một
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
                        + pynutil.insert(linh_word)
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
                        + pynutil.insert(linh_word)
                        + insert_space
                        + self.linh_digits,
                    ]
                )
            elif name == "billion" and digits == 10:
                # Handle patterns like 1001001001
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
