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

        self.two_digit = pynini.union(
            self.teen,
            self.ties + pynutil.delete("0"),
            self.ties
            + insert_space
            + pynini.union(self.special_digits, pynini.union("2", "3", "6", "7", "8", "9") @ self.digit),
        )

        hundred_word = self.magnitudes["hundred"]
        linh_word = self.magnitudes["linh"]

        # X00: một trăm, hai trăm, etc.
        hundreds_exact = self.digit + insert_space + pynutil.insert(hundred_word) + pynutil.delete("00")

        # X0Y: một trăm linh một, hai trăm linh năm, etc.
        hundreds_with_linh = (
            self.digit
            + insert_space
            + pynutil.insert(hundred_word)
            + pynutil.delete("0")
            + insert_space
            + pynutil.insert(linh_word)
            + insert_space
            + self.linh_digits
        )

        # XYZ: một trăm hai mười ba, etc.
        hundreds_with_tens = self.digit + insert_space + pynutil.insert(hundred_word) + insert_space + self.two_digit

        # 0YZ: Handle numbers starting with 0 (e.g., 087 -> tám mươi bảy)
        leading_zero_tens = pynutil.delete("0") + self.two_digit

        # 00Z: Handle numbers starting with 00 (e.g., 008 -> tám)
        leading_double_zero = pynutil.delete("00") + self.digit

        self.hundreds_pattern = pynini.union(
            hundreds_exact,
            hundreds_with_linh,
            hundreds_with_tens,
            leading_zero_tens,
            leading_double_zero,
        )

        self.hundreds = pynini.closure(NEMO_DIGIT, 3, 3) @ self.hundreds_pattern

        self.magnitude_patterns = self._build_all_magnitude_patterns()
        custom_patterns = self._build_all_patterns()

        all_patterns = [
            *custom_patterns,
            *self.magnitude_patterns.values(),
            self.hundreds,
            self.two_digit,
            self.digit,
            self.zero,
        ]
        self.graph = pynini.union(*all_patterns).optimize()

        self.single_digits_graph = self.digit | self.zero

        negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = negative + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph).optimize()

    def _build_magnitude_pattern(self, name, min_digits, max_digits, zero_count, prev_pattern=None):
        magnitude_word = self.magnitudes[name]
        linh_word = self.magnitudes["linh"]
        patterns = []

        for digits in range(min_digits, max_digits + 1):
            leading_digits = digits - zero_count
            if leading_digits == 1:
                leading_fst = self.digit
            elif leading_digits == 2:
                leading_fst = self.two_digit
            else:
                leading_fst = self.hundreds_pattern

            prefix = leading_fst + insert_space + pynutil.insert(magnitude_word)
            digit_patterns = [prefix + pynutil.delete("0" * zero_count)]

            if prev_pattern and name not in ["quadrillion", "quintillion"]:
                digit_patterns.append(prefix + insert_space + prev_pattern)

            for trailing_zeros in range(zero_count):
                remaining_digits = zero_count - trailing_zeros
                trailing_prefix = prefix + pynutil.delete("0" * trailing_zeros)

                if remaining_digits == 1:
                    linh_pattern = (
                        trailing_prefix + insert_space + pynutil.insert(linh_word) + insert_space + self.linh_digits
                    )
                    digit_patterns.append(pynutil.add_weight(linh_pattern, -0.1))
                elif remaining_digits == 2:
                    digit_patterns.append(trailing_prefix + insert_space + self.two_digit)
                elif remaining_digits == 3:
                    digit_patterns.append(trailing_prefix + insert_space + self.hundreds_pattern)

            patterns.append(pynini.closure(NEMO_DIGIT, digits, digits) @ pynini.union(*digit_patterns))

        return pynini.union(*patterns)

    def _build_all_magnitude_patterns(self):
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
            if name in self.magnitudes:
                patterns[name] = self._build_magnitude_pattern(name, min_digits, max_digits, zero_count, prev_pattern)
                prev_pattern = patterns[name]
            else:
                break
        return patterns

    def _get_zero_or_magnitude_pattern(self, digits, magnitude_key):
        """Create pattern that handles all-zeros or normal magnitude processing"""
        all_zeros = "0" * digits
        return pynini.union(pynini.cross(all_zeros, ""), NEMO_DIGIT**digits @ self.magnitude_patterns[magnitude_key])

    def _build_all_patterns(self):
        patterns = []
        delete_dot = pynutil.delete(".")

        # Large number split patterns (>12 digits): front + "tỷ" + back(9 digits)
        if "billion" in self.magnitudes:
            billion_word = self.magnitudes["billion"]
            back_digits = 9

            for total_digits in range(13, 22):
                front_digits = total_digits - back_digits
                front_pattern = self._get_pattern_for_digits(front_digits)
                if front_pattern:
                    back_pattern = self._get_zero_or_magnitude_pattern(back_digits, "million")
                    split_pattern = (
                        front_pattern + insert_space + pynutil.insert(billion_word) + insert_space + back_pattern
                    )
                    patterns.append(NEMO_DIGIT**total_digits @ pynutil.add_weight(split_pattern, -0.5))

        # Dot patterns
        dot_configs = [(6, None), (5, None), (4, None), (3, "billion"), (2, "million"), (1, "thousand")]
        for dots, magnitude in dot_configs:
            pattern = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0, 2)
            for _ in range(dots):
                pattern += delete_dot + NEMO_DIGIT**3

            if magnitude and magnitude in self.magnitude_patterns:
                patterns.append(pynini.compose(pynutil.add_weight(pattern, -0.3), self.magnitude_patterns[magnitude]))
            elif not magnitude:
                if dots == 4:
                    digit_range = [13, 14, 15]
                elif dots == 5:
                    digit_range = [16, 17, 18]
                elif dots == 6:
                    digit_range = [19, 20, 21]
                else:
                    digit_range = []

                for digit_count in digit_range:
                    if 13 <= digit_count <= 21:
                        front_digits = digit_count - back_digits
                        front_pattern = self._get_pattern_for_digits(front_digits)
                        if front_pattern:
                            back_pattern = self._get_zero_or_magnitude_pattern(back_digits, "million")
                            split = (
                                (NEMO_DIGIT**front_digits @ front_pattern)
                                + insert_space
                                + pynutil.insert(self.magnitudes["billion"])
                                + insert_space
                                + back_pattern
                            )
                            patterns.append(
                                pynini.compose(pattern, NEMO_DIGIT**digit_count @ pynutil.add_weight(split, -1.0))
                            )

        return patterns

    def _get_pattern_for_digits(self, digit_count):
        if digit_count <= 0:
            return None
        elif digit_count == 1:
            return self.digit
        elif digit_count == 2:
            return self.two_digit
        elif digit_count == 3:
            return self.hundreds_pattern
        elif digit_count <= 6:
            return self.magnitude_patterns.get("thousand")
        elif digit_count <= 9:
            return self.magnitude_patterns.get("million")
        elif digit_count <= 12:
            return self.magnitude_patterns.get("billion")
        else:
            return None
