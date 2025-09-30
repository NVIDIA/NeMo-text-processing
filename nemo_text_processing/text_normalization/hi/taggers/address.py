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

def get_context(keywords):
    all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)
    
    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
    
    window = pynini.closure(word, 0, 5)
    
    # Make context mandatory by using (1, 1) closure - exactly one occurrence required
    before = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 1, 1)
    after = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 1, 1)
    
    # Also create optional versions for when we need to allow empty context
    before_optional = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1)
    after_optional = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)
    
    return before.optimize(), after.optimize(), before_optional.optimize(), after_optional.optimize()


class AddressFst(GraphFst):
    """
    Finite state transducer for tagging address, e.g.
    """

    def __init__(self):
        super().__init__(name="address", kind="classify")
        
        # Load digit mappings
        digit_to_word = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT
        digit_verbalizer = pynini.compose(single_digit, digit_to_word)

        # Create number sequence with proper spacing (no trailing space)
        number_sequence = pynini.closure(digit_verbalizer + insert_space, 1) + digit_verbalizer

        # Create handlers for different character types
        # English letters: consecutive letters stay together, no internal spaces
        english_letters = pynini.closure(NEMO_ALPHA, 1)
        
        # Digits: each digit becomes a word with spaces between them  
        digit_sequence = pynini.closure(digit_verbalizer + insert_space, 1) + digit_verbalizer
        
        # Alphanumeric components (letters OR digits)
        alphanumeric_component = english_letters | digit_sequence
        
        # Alphanumeric sequence: components separated by spaces
        alphanumeric_with_digit = pynini.closure(alphanumeric_component + insert_space, 1) + alphanumeric_component

        context_before, context_after, context_before_optional, context_after_optional = get_context(address_context)
        
        # Create a version of context_after that inserts space when present (mandatory)
        all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)
        non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
        word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
        window = pynini.closure(word, 0, 5)
        context_after_with_space = pynini.closure(insert_space + pynutil.delete(NEMO_SPACE) + window + address_context, 1, 1)
        
        # Create optional version for when context_before is present
        context_after_with_space_optional = pynini.closure(insert_space + pynutil.delete(NEMO_SPACE) + window + address_context, 0, 1)
        
        # Build the graph with mandatory context requirements
        # Handle both pure numeric and alphanumeric patterns
        numeric_patterns = (
            (context_before + number_sequence + context_after_with_space_optional) |
            (context_before_optional + number_sequence + context_after_with_space)
        )
        
        alphanumeric_patterns = (
            (context_before + alphanumeric_with_digit + context_after_with_space_optional) |
            (context_before_optional + alphanumeric_with_digit + context_after_with_space)
        )
        
        graph = numeric_patterns | alphanumeric_patterns
        
        final_graph = pynutil.insert('number_part: "') + graph + pynutil.insert('"')
        final_graph = pynutil.add_weight(final_graph, -0.1)
        self.fst = self.add_tokens(final_graph)