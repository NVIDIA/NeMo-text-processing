# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    GraphFst,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    NEMO_ALPHA,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

address_context = pynini.string_file(get_abs_path("data/address/address_context.tsv"))

# Special character mappings for addresses
hyphen_mapping = pynini.cross("-", "हाइफ़न")
slash_mapping = pynini.cross("/", "बटा")

def get_context(keywords):
    all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)
    
    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
    
    window = pynini.closure(word, 0, 5)
    
    # Create flexible context patterns that allow context in either direction
    before = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1)
    after = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)
    
    # Create mandatory context requirement: at least one context keyword must be present
    # This allows either before OR after context (or both)
    context_present = (
        pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 1, 1) + 
        pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)
    ) | (
        pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1) + 
        pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 1, 1)
    )
    
    return before.optimize(), after.optimize(), context_present.optimize(), after.optimize()


class AddressFst(GraphFst):
    """
    Finite state transducer for tagging address patterns with digit-by-digit conversion,
    special character handling (hyphen -> हाइफ़न, slash -> बटा), and address-specific patterns.
    """

    def __init__(self):
        super().__init__(name="address", kind="classify")
        
        # Load digit mappings for digit-by-digit conversion
        digit_to_word = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        # Define character sets
        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT
        digit_verbalizer = pynini.compose(single_digit, digit_to_word)
        
        # Create digit-by-digit number sequence with proper spacing
        digit_sequence = pynini.closure(digit_verbalizer + insert_space, 1) + digit_verbalizer
        single_digit_verbalized = digit_verbalizer
        
        # Create number sequence (same as digit_sequence for consistency)
        number_sequence = digit_sequence
        
        # Non-digit characters for text parts (excluding hyphen and slash for proper pattern separation)
        special_chars = pynini.union("-", "/")
        non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(single_digit, NEMO_WHITE_SPACE, special_chars))
        text_part = pynini.closure(non_digit_char, 1)
        
        # Text part that can include special characters (for patterns that handle them)
        non_digit_char_with_specials = pynini.difference(NEMO_CHAR, pynini.union(single_digit, NEMO_WHITE_SPACE))
        text_part_with_specials = pynini.closure(non_digit_char_with_specials, 1)
        
        # Create handlers for different character types
        # English letters: consecutive letters stay together, no internal spaces
        english_letters = pynini.closure(NEMO_ALPHA, 1)
        
        # Alphanumeric character (allows A-Z)
        alpha_char = pynini.union(NEMO_ALPHA, pynini.accep("A"), pynini.accep("B"), pynini.accep("C"), 
                                 pynini.accep("D"), pynini.accep("E"), pynini.accep("F"), pynini.accep("G"),
                                 pynini.accep("H"), pynini.accep("I"), pynini.accep("J"), pynini.accep("K"),
                                 pynini.accep("L"), pynini.accep("M"), pynini.accep("N"), pynini.accep("O"),
                                 pynini.accep("P"), pynini.accep("Q"), pynini.accep("R"), pynini.accep("S"),
                                 pynini.accep("T"), pynini.accep("U"), pynini.accep("V"), pynini.accep("W"),
                                 pynini.accep("X"), pynini.accep("Y"), pynini.accep("Z"))
        
        # Alphanumeric components (letters OR digits)
        alphanumeric_component = english_letters | digit_sequence
        
        # Alphanumeric sequence: components separated by spaces
        alphanumeric_with_digit = pynini.closure(alphanumeric_component + insert_space, 1) + alphanumeric_component
        
        # Pattern components
        
        # 1. Simple digit sequences (convert digit by digit)
        simple_digits = digit_sequence
        
        # 2. Text-hyphen-digits patterns (e.g., अन्तर्गत-७३६५५७)
        text_hyphen_digits = (
            text_part + 
            insert_space + pynutil.delete("-") + pynutil.insert("हाइफ़न") + insert_space +
            digit_sequence
        )
        
        # 3. Digits-slash-digits patterns (e.g., ९/६)
        digits_slash_digits = (
            digit_sequence +
            insert_space + pynutil.delete("/") + pynutil.insert("बटा") + insert_space +
            digit_sequence
        )
        
        # 4. Digits-hyphen-digits patterns (e.g., ६६-४)
        digits_hyphen_digits = (
            digit_sequence +
            insert_space + pynutil.delete("-") + pynutil.insert("हाइफ़न") + insert_space +
            digit_sequence
        )
        
        # 5. Alphanumeric patterns (e.g., NH४१, ३२A)
        alpha_digits = (
            pynini.closure(alpha_char, 1) + insert_space + digit_sequence
        ) | (
            digit_sequence + insert_space + pynini.closure(alpha_char, 1)
        )
        
        # Address pattern definitions with context
        
        # Pattern A: Numbers with following text (e.g., ७१ गोविंदा कृष्णा धारवाड)
        digits_with_text = simple_digits + insert_space + text_part + pynini.closure(insert_space + text_part, 0)
        
        # Pattern B: Text with following numbers (e.g., चक्रधरपुर ७३६५५७)
        text_with_digits = text_part + insert_space + simple_digits
        
        # Pattern C: Text-hyphen-digits with optional following text
        text_hyphen_digits_with_context = text_hyphen_digits + pynini.closure(insert_space + text_part, 0)
        
        # Pattern D: Digits-slash-digits with optional context
        digits_slash_digits_with_context = digits_slash_digits + pynini.closure(insert_space + text_part, 0)
        
        # Pattern E: Digits-hyphen-digits with following text
        digits_hyphen_digits_with_context = digits_hyphen_digits + pynini.closure(insert_space + text_part, 0)
        
        # Pattern F: Alphanumeric with following text
        alpha_digits_with_context = alpha_digits + pynini.closure(insert_space + text_part, 0)
        
        # Pattern G: Multi-digit sequences in various contexts (for pin codes etc.)
        long_digit_sequence = pynini.compose(
            pynini.closure(single_digit, 3),  # At least 3 digits
            digit_sequence
        )
        long_digits_with_context = (
            long_digit_sequence + pynini.closure(insert_space + text_part, 0)
        ) | (
            text_part + insert_space + long_digit_sequence
        )
        
        # Pattern H: Text-hyphen-digits followed by text in middle of sentence
        text_hyphen_digits_text = text_part + insert_space + pynutil.delete("-") + pynutil.insert("हाइफ़न") + insert_space + digit_sequence + insert_space + text_part
        
        # Pattern I: Complex address patterns (text-digits-text combinations)
        complex_address = (
            text_part + insert_space + digit_sequence + insert_space + text_part + 
            pynini.closure(insert_space + text_part, 0)
        )
        
        # Special priority patterns for common address issues
        
        # Handle specific 2-digit cases that are being missed (high priority)
        two_digit_with_text = (
            pynini.compose(
                pynini.closure(single_digit, 2, 2),  # Exactly 2 digits 
                digit_sequence
            ) + insert_space + text_part + pynini.closure(insert_space + text_part, 0)
        )
        
        # Single digit with text (to catch edge cases)
        single_digit_with_text = (
            single_digit_verbalized + insert_space + text_part + pynini.closure(insert_space + text_part, 0)
        )
        
        # Text followed by 2-digit number (like "के २८")
        text_two_digit = (
            text_part + insert_space + 
            pynini.compose(
                pynini.closure(single_digit, 2, 2),  # Exactly 2 digits
                digit_sequence
            ) + insert_space + text_part + pynini.closure(insert_space + text_part, 0)
        )
        
        # Handle range patterns like ६६-४ 
        range_pattern = (
            pynini.compose(
                pynini.closure(single_digit, 1, 2),  # 1-2 digits
                digit_sequence
            ) + 
            insert_space + pynutil.delete("-") + pynutil.insert("हाइफ़न") + insert_space +
            pynini.compose(
                pynini.closure(single_digit, 1, 2),  # 1-2 digits
                digit_sequence
            ) + insert_space + text_part + pynini.closure(insert_space + text_part, 0)
        )
        
        # Text-hyphen-digits in middle of text (like हबर्ड एवेन्यू-स्वीट १२)
        text_hyphen_text = (
            text_part + insert_space + text_part + 
            insert_space + pynutil.delete("-") + pynutil.insert("हाइफ़न") + insert_space + 
            digit_sequence + pynini.closure(insert_space + text_part, 0)
        )
        
        # Combine all address patterns with hyphen/slash patterns FIRST for highest priority
        address_patterns = (
            # Hyphen and slash patterns get highest priority
            text_hyphen_digits_with_context |
            digits_hyphen_digits_with_context |
            text_hyphen_text |
            digits_slash_digits_with_context |
            range_pattern |
            # Then other specific patterns
            alpha_digits_with_context |
            text_hyphen_digits |  # Direct hyphen pattern
            digits_slash_digits |  # Direct slash pattern  
            digits_hyphen_digits |  # Direct hyphen-digit pattern
            two_digit_with_text |
            single_digit_with_text |
            text_two_digit |
            digits_with_text |
            text_with_digits |
            long_digits_with_context |
            complex_address |
            # Basic patterns last so they don't override specific patterns
            alphanumeric_with_digit |
            number_sequence
        )

        # Apply STRICT context requirement to prevent interfering with cardinal tagger
        # Context is now MANDATORY - address patterns can only match when address keywords are present
        all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)
        non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
        word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
        window = pynini.closure(word, 0, 5)
        
        # MANDATORY context: require at least one address keyword before OR after the pattern
        context_before = pynini.closure(address_context + pynini.accep(NEMO_SPACE) + window, 1, 1)
        context_after = pynini.closure(insert_space + pynutil.delete(NEMO_SPACE) + window + address_context, 1, 1)
        
        # Either context before OR context after is required (not optional)
        graph = (
            (context_before + address_patterns + pynini.closure(insert_space + pynutil.delete(NEMO_SPACE) + window + address_context, 0, 1)) |
            (pynini.closure(address_context + pynini.accep(NEMO_SPACE) + window, 0, 1) + address_patterns + context_after)
        )
        
        # Use proven weight from working approach
        final_graph = pynutil.insert('number_part: "') + graph + pynutil.insert('"')
        final_graph = pynutil.add_weight(final_graph, -0.1)
        self.fst = self.add_tokens(final_graph)