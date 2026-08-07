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
    NEMO_COMMA,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "10,5$" -> money { integer_part: "mười" currency_maj: "đô la" fractional_part: "năm mươi" currency_min: "xu" preserve_order: true }
        "10đ" -> money { integer_part: "mười" currency_maj: "đồng" }
        "10 triệu đồng" -> money { integer_part: "mười" quantity: "triệu" currency_maj: "đồng" }

    Args:
        cardinal: CardinalFst instance for processing integer parts
        decimal: DecimalFst instance for processing fractional parts
        deterministic: if True will provide a single transduction option, for False multiple transduction are generated.
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        # Load data
        currency_major_labels = load_labels(get_abs_path("data/money/currency.tsv"))
        currency_minor_labels = load_labels(get_abs_path("data/money/currency_minor.tsv"))
        quantity_graph = pynini.string_file(get_abs_path("data/numbers/quantity_abbr.tsv"))

        # Load optimized per_unit files using subfst approach
        per_unit_non_metric_path = get_abs_path("data/money/per_unit_non_metric.tsv")
        per_unit_prefixes_path = get_abs_path("data/money/per_unit_prefixes.tsv")
        per_unit_bases_path = get_abs_path("data/money/per_unit_bases.tsv")

        # Create subfst for metric per_unit patterns
        graph_prefixes = pynini.string_file(per_unit_prefixes_path)
        graph_bases = pynini.string_file(per_unit_bases_path)

        # Build metric combinations: "/kg" -> "một ki lô gam"
        slash = pynutil.delete("/")
        one_space = pynutil.insert("một ")
        space = pynutil.insert(NEMO_SPACE)

        graph_metric_per_units = slash + one_space + graph_prefixes + space + graph_bases
        graph_standalone_per_units = slash + one_space + graph_bases

        # Load non-metric per_unit entries
        graph_non_metric_per_units = pynini.string_file(per_unit_non_metric_path)

        # Combine all per_unit mappings
        per_unit_graph = graph_metric_per_units | graph_standalone_per_units | graph_non_metric_per_units

        # Basic components
        cardinal_graph = cardinal.graph
        currency_major_graph = pynini.string_map(currency_major_labels)
        currency_minor_map = dict(currency_minor_labels)
        decimal_graph = decimal.final_graph_wo_negative

        # Common patterns
        integer_part = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        preserve_order = pynutil.insert(" preserve_order: true")
        optional_space = pynini.closure(delete_space, 0, 1)

        # Fractional part conversion for cents
        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )
        fractional_conversion = two_digits_fractional_part @ cardinal_graph
        fractional_part = pynutil.insert('fractional_part: "') + fractional_conversion + pynutil.insert('"')

        all_patterns = []

        # 1. Symbol-based patterns
        symbol_patterns = []
        minor_only_patterns = []

        for symbol, major_name in currency_major_labels:
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')

            # Simple integer pattern: 10$ -> mười đô la
            simple_pattern = integer_part + pynutil.delete(symbol) + insert_space + maj_tag
            symbol_patterns.append(simple_pattern)

            # Patterns with minor currency (cents/xu)
            if symbol in currency_minor_map:
                minor_name = currency_minor_map[symbol]
                min_tag = pynutil.insert(f' currency_min: "{minor_name}"')

                # Minor-only pattern: 0,5$ -> năm mươi xu (highest priority)
                minor_only = (
                    pynutil.delete("0")
                    + pynutil.delete(NEMO_COMMA)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + pynutil.delete(symbol)
                    + preserve_order
                )
                minor_only_patterns.append(minor_only)

                # Major + minor pattern: 10,5$ -> mười đô la năm mươi xu
                major_minor = (
                    integer_part
                    + insert_space
                    + maj_tag
                    + pynini.cross(NEMO_COMMA, NEMO_SPACE)
                    + fractional_part
                    + insert_space
                    + min_tag
                    + pynutil.delete(symbol)
                    + preserve_order
                )
                symbol_patterns.append(major_minor)

        # 2. Word-based patterns
        word_patterns = []

        # Complex decimal + currency: 1tr5 vnd -> một triệu năm trăm nghìn đồng
        decimal_with_currency = (
            decimal_graph
            + optional_space
            + insert_space
            + pynutil.insert(' currency_maj: "')
            + convert_space(currency_major_graph)
            + pynutil.insert('"')
        )
        word_patterns.append(decimal_with_currency)

        # Quantity + currency: 10tr đồng -> mười triệu đồng
        quantity_tag = pynutil.insert(' quantity: "') + convert_space(quantity_graph) + pynutil.insert('"')
        quantity_pattern = (
            integer_part
            + optional_space
            + insert_space
            + quantity_tag
            + optional_space
            + insert_space
            + pynutil.insert(' currency_maj: "')
            + convert_space(currency_major_graph)
            + pynutil.insert('"')
        )
        word_patterns.append(quantity_pattern)

        # Simple word pattern: 10 đồng -> mười đồng
        simple_word_pattern = (
            integer_part
            + optional_space
            + insert_space
            + pynutil.insert(' currency_maj: "')
            + convert_space(currency_major_graph)
            + pynutil.insert('"')
        )
        word_patterns.append(simple_word_pattern)

        # Combine patterns with priorities
        # Minor-only patterns get highest priority (negative weight)
        if minor_only_patterns:
            all_patterns.append(pynutil.add_weight(pynini.union(*minor_only_patterns), -0.0001))

        # Symbol patterns get normal priority
        if symbol_patterns:
            all_patterns.append(pynini.union(*symbol_patterns))

        # Word patterns get lowest priority
        if word_patterns:
            all_patterns.append(pynutil.add_weight(pynini.union(*word_patterns), 0.1))

        # Final graph with optional per-unit support
        final_graph = pynini.union(*all_patterns)
        per_unit_tag = pynutil.insert(' morphosyntactic_features: "') + per_unit_graph + pynutil.insert('"')
        final_graph += per_unit_tag.ques

        self.fst = self.add_tokens(final_graph.optimize())
