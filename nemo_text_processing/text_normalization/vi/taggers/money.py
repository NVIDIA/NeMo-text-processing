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
    NEMO_DIGIT,
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
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        
        # Load data from files
        currency_major_labels = load_labels(get_abs_path("data/money/currency.tsv"))
        currency_minor_labels = load_labels(get_abs_path("data/money/currency_minor.tsv"))
        quantity_graph = pynini.string_file(get_abs_path("data/numbers/quantity_abbr.tsv"))
        per_unit_graph = pynini.string_file(get_abs_path("data/money/per_unit.tsv"))
        
        # Core infrastructure
        cardinal_graph = cardinal.graph
        currency_major_graph = pynini.string_map(currency_major_labels)
        currency_minor_map = dict(currency_minor_labels)
        
        # Building blocks
        integer_part = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        preserve_order = pynutil.insert(" preserve_order: true")
        optional_space = pynini.closure(delete_space, 0, 1)
        
        # Vietnamese cent conversion for fractional parts
        # Convert fractional digits to proper Vietnamese numbers
        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )
        
        fractional_conversion = two_digits_fractional_part @ cardinal_graph
        fractional_part = pynutil.insert('fractional_part: "') + fractional_conversion + pynutil.insert('"')
        
        # Build symbol-based currency patterns
        symbol_patterns = []
        minor_patterns = []  # Separate collection for minor-only patterns
        
        for symbol, major_name in currency_major_labels:
            maj_tag = pynutil.insert(f' currency_maj: "{major_name}"')
            
            if symbol in currency_minor_map:
                minor_name = currency_minor_map[symbol]
                min_tag = pynutil.insert(f' currency_min: "{minor_name}"')
                
                # Pattern 1: Minor only (0,5$ -> năm mươi xu) - collect separately for priority
                minor_only = (
                    pynutil.delete("0,") + fractional_part + insert_space + min_tag + 
                    pynutil.delete(symbol) + preserve_order
                )
                minor_patterns.append(minor_only)  # Add to separate collection
                
                # Pattern 2: Major + minor (10,5$ -> mười đô la năm mươi xu) - lower priority than minor-only
                major_minor = pynutil.add_weight(
                    integer_part + insert_space + maj_tag + 
                    pynini.cross(",", " ") + fractional_part + insert_space + min_tag + 
                    pynutil.delete(symbol) + preserve_order,
                    0.0001  # Positive weight = lower priority than minor-only (-0.0001)
                )
                symbol_patterns.append(major_minor)
            
            # Pattern 3: Simple integer (10$ -> mười đô la) - normal priority
            simple_integer = (
                integer_part + pynutil.delete(symbol) + insert_space + maj_tag
            )
            symbol_patterns.append(simple_integer)
        
        # Word-based currency patterns (lower priority)
        word_patterns = []
        
        # Decimal + currency word patterns: 1tr5 vnd -> một triệu năm trăm nghìn đồng
        # Use the decimal graph (without negative) to handle complex number patterns
        decimal_graph = decimal.final_graph_wo_negative
        decimal_with_currency = (
            decimal_graph + optional_space + insert_space + pynutil.insert(' currency_maj: "') + 
            convert_space(currency_major_graph) + pynutil.insert('"')
        )
        word_patterns.append(decimal_with_currency)
        
        # Quantity + currency: 10tr đồng -> mười triệu đồng
        quantity_tag = pynutil.insert(' quantity: "') + convert_space(quantity_graph) + pynutil.insert('"')
        quantity_pattern = (
            integer_part + optional_space + insert_space + quantity_tag + 
            optional_space + insert_space + pynutil.insert(' currency_maj: "') + 
            convert_space(currency_major_graph) + pynutil.insert('"')
        )
        word_patterns.append(quantity_pattern)
        
        # Simple currency word: 10 đồng -> mười đồng  
        simple_word_pattern = (
            integer_part + optional_space + insert_space + pynutil.insert(' currency_maj: "') + 
            convert_space(currency_major_graph) + pynutil.insert('"')
        )
        word_patterns.append(simple_word_pattern)
        
        # Combine patterns with explicit priorities (following English approach)
        final_graph = None
        
        # Step 1: Start with symbol patterns (normal priority)
        if symbol_patterns:
            final_graph = pynini.union(*symbol_patterns)
        
        # Step 2: Add minor-only patterns with HIGHEST priority (negative weight)
        if minor_patterns:
            minor_graph = pynini.union(*minor_patterns)
            if final_graph is None:
                final_graph = pynutil.add_weight(minor_graph, -0.0001)
            else:
                final_graph |= pynutil.add_weight(minor_graph, -0.0001)  # Highest priority
        
        # Step 3: Add word patterns (lowest priority)
        if word_patterns:
            word_graph = pynini.union(*word_patterns)
            if final_graph is None:
                final_graph = pynutil.add_weight(word_graph, 0.1)
            else:
                final_graph |= pynutil.add_weight(word_graph, 0.1)
        
        # Add per-unit support
        per_unit_tag = pynutil.insert(' morphosyntactic_features: "') + per_unit_graph + pynutil.insert('"')
        final_graph += per_unit_tag.ques
        
        # Finalize
        self.fst = self.add_tokens(final_graph.optimize()) 