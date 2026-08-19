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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_COMMA, NEMO_DIGIT, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese decimal numbers, e.g.
        -12,5 tỷ -> decimal { negative: "true" integer_part: "mười hai" fractional_part: "năm" quantity: "tỷ" }
        12.345,67 -> decimal { integer_part: "mười hai nghìn ba trăm bốn mươi lăm" fractional_part: "sáu bảy" }
        1tr2 -> decimal { integer_part: "một triệu hai trăm nghìn" }
        818,303 -> decimal { integer_part: "tám trăm mười tám" fractional_part: "ba không ba" }
        0,2 triệu -> decimal { integer_part: "không" fractional_part: "hai" quantity: "triệu" }
    Args:
        cardinal: CardinalFst instance for processing integer parts
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        self.graph = cardinal.single_digits_graph.optimize()
        if not deterministic:
            self.graph = self.graph | cardinal_graph

        # Load data
        digit_labels = load_labels(get_abs_path("data/numbers/digit.tsv"))
        zero_labels = load_labels(get_abs_path("data/numbers/zero.tsv"))
        magnitude_labels = load_labels(get_abs_path("data/numbers/magnitudes.tsv"))
        quantity_abbr_labels = load_labels(get_abs_path("data/numbers/quantity_abbr.tsv"))

        # Common components
        single_digit_map = pynini.union(*[pynini.cross(k, v) for k, v in digit_labels + zero_labels])
        quantity_units = pynini.union(*[v for _, v in magnitude_labels])
        one_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        # Building blocks
        integer_part = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        fractional_part = (
            pynutil.insert("fractional_part: \"")
            + single_digit_map
            + pynini.closure(pynutil.insert(NEMO_SPACE) + single_digit_map)
            + pynutil.insert("\"")
        )
        optional_quantity = (
            pynutil.delete(NEMO_SPACE).ques + pynutil.insert(" quantity: \"") + quantity_units + pynutil.insert("\"")
        ).ques

        patterns = []

        # 1. Basic decimal patterns: 12,5 and 12,5 tỷ
        basic_decimal = (
            (integer_part + pynutil.insert(NEMO_SPACE)).ques
            + pynutil.delete(NEMO_COMMA)
            + pynutil.insert(NEMO_SPACE)
            + fractional_part
        )
        patterns.append(basic_decimal)
        patterns.append(basic_decimal + optional_quantity)

        # 2. Thousand-separated decimals: 12.345,67 and 12.345,67 tỷ
        integer_with_dots = (
            NEMO_DIGIT + pynini.closure(NEMO_DIGIT, 0, 2) + pynini.closure(pynutil.delete(".") + NEMO_DIGIT**3, 1)
        )
        separated_integer_part = (
            pynutil.insert("integer_part: \"")
            + pynini.compose(integer_with_dots, cardinal_graph)
            + pynutil.insert("\"")
        )
        separated_decimal = (
            separated_integer_part
            + pynutil.insert(NEMO_SPACE)
            + pynutil.delete(NEMO_COMMA)
            + pynutil.insert(NEMO_SPACE)
            + fractional_part
        )
        patterns.append(separated_decimal)
        patterns.append(separated_decimal + optional_quantity)

        # 3. Integer with quantity: 100 triệu
        integer_with_quantity = (
            integer_part
            + pynutil.delete(NEMO_SPACE).ques
            + pynutil.insert(" quantity: \"")
            + quantity_units
            + pynutil.insert("\"")
        )
        patterns.append(integer_with_quantity)

        # 4. Standard abbreviations: 1k, 100tr, etc.
        for abbr, full_name in quantity_abbr_labels:
            abbr_pattern = pynini.compose(
                one_to_three_digits + pynutil.delete(abbr),
                pynutil.insert("integer_part: \"")
                + pynini.compose(one_to_three_digits, cardinal_graph)
                + pynutil.insert(f"\" quantity: \"{full_name}\""),
            )
            patterns.append(abbr_pattern)

        # 5. Decimal with abbreviations: 2,5tr, but avoid measure conflicts
        measure_prefix_labels = load_labels(get_abs_path("data/measure/prefixes.tsv"))
        measure_prefixes = {prefix.lower() for prefix, _ in measure_prefix_labels}

        # Filter quantity abbreviations to avoid measure conflicts
        safe_quantity_abbrs = [
            (abbr, full) for abbr, full in quantity_abbr_labels if abbr.lower() not in measure_prefixes
        ]

        for abbr, full_name in safe_quantity_abbrs:
            decimal_abbr_pattern = (
                (integer_part + pynutil.insert(NEMO_SPACE)).ques
                + pynutil.delete(NEMO_COMMA)
                + pynutil.insert(NEMO_SPACE)
                + fractional_part
                + pynutil.insert(f" quantity: \"{full_name}\"")
                + pynutil.delete(abbr)
            )
            patterns.append(decimal_abbr_pattern)

        # 6. Compound abbreviations: 1tr2 -> một triệu hai trăm nghìn, 2t3 -> hai tỷ ba trăm triệu
        compound_expansions = {
            "tr": ("triệu", "trăm nghìn"),  # 1tr2 -> một triệu hai trăm nghìn
            "t": ("tỷ", "trăm triệu"),  # 2t3 -> hai tỷ ba trăm triệu
        }

        for abbr, (major_unit, minor_suffix) in compound_expansions.items():
            pattern = one_to_three_digits + pynini.cross(abbr, "") + NEMO_DIGIT
            expansion = (
                pynutil.insert("integer_part: \"")
                + pynini.compose(one_to_three_digits, cardinal_graph)
                + pynutil.insert(f" {major_unit} ")
                + pynini.compose(NEMO_DIGIT, cardinal_graph)
                + pynutil.insert(f" {minor_suffix}\"")
            )
            patterns.append(pynini.compose(pattern, expansion))

        # Combine all patterns
        self._final_graph_wo_negative = pynini.union(*patterns).optimize()

        # Add optional negative prefix
        negative = (pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")).ques
        final_graph = negative + self._final_graph_wo_negative

        self.fst = self.add_tokens(final_graph).optimize()

    @property
    def final_graph_wo_negative(self):
        """Graph without negative prefix, used by MoneyFst"""
        return self._final_graph_wo_negative
