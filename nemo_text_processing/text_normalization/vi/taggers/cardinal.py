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
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


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

        magnitudes_labels = load_labels(get_abs_path("data/numbers/magnitudes.tsv"))
        self.magnitudes = {parts[0]: parts[1] for parts in magnitudes_labels if len(parts) == 2}

        digit_special_labels = load_labels(get_abs_path("data/numbers/digit_special.tsv"))
        special = {parts[0]: {'std': parts[1], 'alt': parts[2]} for parts in digit_special_labels if len(parts) >= 3}

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
            # 0YZ: Handle numbers starting with 0 (e.g., 087 -> tám mươi bảy)
            pynutil.delete("0") + self.two_digit,
            # 00Z: Handle numbers starting with 00 (e.g., 008 -> tám)
            pynutil.delete("00") + self.single_digit,
        )

        self.hundreds = pynini.closure(NEMO_DIGIT, 3, 3) @ self.hundreds_pattern

        # Build magnitude patterns dynamically
        self.magnitude_patterns = self._build_all_magnitude_patterns()

        # Handle dot-separated numbers: 1.000, 1.000.000, etc.
        delete_dot = pynutil.delete(".")
        dot_patterns = []

        # Build dot patterns automatically for all available magnitudes
        for i, magnitude_name in enumerate(
            ["thousand", "million", "billion", "trillion", "quadrillion", "quintillion"], 1
        ):
            if magnitude_name in self.magnitude_patterns:
                # Build pattern: (non-zero digit) + up to 2 digits + (dot + 3 digits) repeated i times
                pattern = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0, 2)
                for _ in range(i):  # i = number of dot groups for this magnitude
                    pattern += delete_dot + NEMO_DIGIT ** 3

                dot_patterns.append(pynini.compose(pattern, self.magnitude_patterns[magnitude_name]))

        # Build final graph with all magnitude patterns
        all_patterns = [
            *self.magnitude_patterns.values(),  # All magnitude patterns (trillion, billion, million, thousand)
            self.hundreds,
            self.two_digit,
            self.single_digit,
            self.zero,
            *dot_patterns,
        ]
        self.graph = pynini.union(*all_patterns).optimize()

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
                    # Prefer "linh" pattern with better weight
                    linh_pattern = (
                        trailing_prefix + insert_space + pynutil.insert(linh_word) + insert_space + self.linh_digits
                    )
                    digit_patterns.append(pynutil.add_weight(linh_pattern, -0.1))
                elif remaining_digits == 2:
                    digit_patterns.append(trailing_prefix + insert_space + self.two_digit)
                elif remaining_digits == 3:
                    digit_patterns.append(trailing_prefix + insert_space + self.hundreds_pattern)

            # Handle special cross-magnitude patterns (e.g., 1001001 -> một triệu một nghìn linh một)
            if name == "million" and digits == 7 and "thousand" in self.magnitudes:
                # Use helper method to build linh patterns consistently
                digit_patterns.extend(
                    [
                        self._build_linh_pattern(prefix, 2, self.magnitudes["thousand"], linh_word, self.single_digit),
                        self._build_linh_pattern(prefix, 1, self.magnitudes["thousand"], linh_word, self.two_digit),
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

    def _build_linh_pattern(self, prefix, zeros_to_delete, magnitude_word, linh_word, digit_pattern):
        """
        Helper method to build linh patterns consistently
        Args:
            prefix: base prefix pattern
            zeros_to_delete: number of zeros to delete (0, 00, etc.)
            magnitude_word: magnitude word to insert
            linh_word: linh word to insert  
            digit_pattern: pattern for the digits (single_digit or two_digit)
        """
        pattern = (
            prefix
            + pynutil.delete("0" * zeros_to_delete)
            + insert_space
            + digit_pattern
            + insert_space
            + pynutil.insert(magnitude_word)
            + pynutil.delete("00")
            + insert_space
            + pynutil.insert(linh_word)
            + insert_space
            + self.linh_digits
        )
        return pynutil.add_weight(pattern, -0.1)

    def _build_all_magnitude_patterns(self):
        """
        Dynamically build all magnitude patterns 
        Returns: dict mapping magnitude names to their FST patterns
        """
        # Define magnitude hierarchy (name, min_digits, max_digits, zero_count)
        magnitude_config = [
            ("thousand", 4, 6, 3),
            ("million", 7, 9, 6),
            ("billion", 10, 12, 9),
            ("trillion", 13, 15, 12),
            ("quadrillion", 16, 18, 15),
            ("quintillion", 19, 21, 18),
        ]

        patterns = {}
        prev_pattern = None

        for name, min_digits, max_digits, zero_count in magnitude_config:
            # Only build pattern if the magnitude word exists in magnitudes.tsv
            if name in self.magnitudes:
                patterns[name] = self._build_magnitude_pattern(name, min_digits, max_digits, zero_count, prev_pattern)
                prev_pattern = patterns[name]
            else:
                # Stop building patterns if magnitude word doesn't exist
                break

        return patterns
