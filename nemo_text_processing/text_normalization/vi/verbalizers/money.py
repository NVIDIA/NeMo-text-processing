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

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_COMMA_VI,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "mười" currency_maj: "đồng" } -> "mười đồng"
        money { integer_part: "mười" quantity: "triệu" currency_maj: "đồng" } -> "mười triệu đồng"
        money { integer_part: "mười" currency_maj: "đô la" fractional_part: "năm mươi" currency_min: "xu" preserve_order: true } -> "mười đô la năm mươi xu"
        money { fractional_part: "năm mươi" currency_min: "xu" preserve_order: true } -> "năm mươi xu"
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        fractional_part = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )
        quantity = pynutil.delete('quantity: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        currency_maj = pynutil.delete('currency_maj: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        currency_min = pynutil.delete('currency_min: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Following English prioritization pattern for better determinism

        # 1. Minor only: fractional + minor (highest priority for fractional-only cases)
        graph_minor = fractional_part + delete_space + insert_space + currency_min + delete_preserve_order

        # 2. Major + minor: integer + major + fractional + minor (for complete cases like 10,5$)
        graph_integer_with_minor = (
            integer_part
            + delete_space
            + insert_space
            + currency_maj
            + delete_space
            + insert_space
            + fractional_part
            + delete_space
            + insert_space
            + currency_min
            + delete_preserve_order
        )

        # 3. Simple integer + currency (most common case)
        graph_integer = integer_part + delete_space + insert_space + currency_maj

        # 4. With quantity: integer + quantity + currency
        graph_with_quantity = (
            integer_part + delete_space + insert_space + quantity + delete_space + insert_space + currency_maj
        )

        # 5. Decimal format (using "phẩy" for comma) - for cases like 10,5 đồng
        graph_decimal = (
            integer_part
            + delete_space
            + insert_space
            + pynutil.insert(NEMO_COMMA_VI)
            + insert_space
            + fractional_part
            + delete_space
            + insert_space
            + currency_maj
        )

        # 6. Decimal with quantity: integer + fractional + quantity + currency - for cases like 2,5 triệu đồng
        graph_decimal_with_quantity = (
            integer_part
            + delete_space
            + insert_space
            + pynutil.insert(NEMO_COMMA_VI)
            + insert_space
            + fractional_part
            + delete_space
            + insert_space
            + quantity
            + delete_space
            + insert_space
            + currency_maj
        )

        # Create main graph with proper priority order (similar to English)
        graph = (
            graph_minor  # Handle minor-only cases first
            | graph_integer_with_minor  # Handle major+minor cases
            | graph_decimal_with_quantity  # Handle decimal with quantity cases (before simpler decimal)
            | graph_with_quantity  # Handle quantity cases
            | graph_decimal  # Handle decimal cases
            | graph_integer  # Handle simple cases (most common, lowest priority)
        )

        per_units_non_metric = pynini.string_file(get_abs_path("data/money/per_unit_non_metric.tsv"))

        per_unit_prefixes = pynini.string_file(get_abs_path("data/money/per_unit_prefixes.tsv"))
        per_unit_bases = pynini.string_file(get_abs_path("data/money/per_unit_bases.tsv"))

        prefixes_vn = pynini.project(per_unit_prefixes, "output")
        bases_vn = pynini.project(per_unit_bases, "output")

        one = pynini.accep("một")

        # Accept metric combinations: "một ki lô gam"
        metric_per_units = one + insert_space + prefixes_vn + insert_space + bases_vn
        standalone_per_units = one + insert_space + bases_vn

        # Combine all per_unit recognitions
        per_units = per_units_non_metric | metric_per_units | standalone_per_units
        per_units_normalized = pynini.project(per_units, "output")
        per_unit_pattern = (
            pynutil.delete(' morphosyntactic_features: "') + insert_space + per_units_normalized + pynutil.delete('"')
        )

        # Optional per-unit suffix
        graph += per_unit_pattern.ques

        # Handle preserve_order deletion (should be last)
        graph += (delete_space + pynutil.delete("preserve_order: true")).ques

        self.fst = self.delete_tokens(graph).optimize()
