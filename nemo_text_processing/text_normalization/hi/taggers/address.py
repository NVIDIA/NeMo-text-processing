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
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

address_context = pynini.string_file(get_abs_path("data/address/address_context.tsv"))

def get_context(keywords):
    all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)
    
    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
    
    window = pynini.closure(word, 0, 5)
    
    before = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1)
    after = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)
    
    return before.optimize(), after.optimize()


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

        context_before, context_after = get_context(address_context)
        
        # Create a version of context_after that inserts space when present
        all_digits = pynini.union(NEMO_HI_DIGIT, NEMO_DIGIT)
        non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
        word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
        window = pynini.closure(word, 0, 5)
        context_after_with_space = pynini.closure(insert_space + pynutil.delete(NEMO_SPACE) + window + address_context, 0, 1)
        
        # Build the graph with context requirements and proper spacing
        # At least one context (before or after) must be present  
        graph = (
            (context_before + number_sequence + context_after_with_space) |
            (context_before + number_sequence) |
            (number_sequence + context_after_with_space)
        )
        
        final_graph = pynutil.insert('number_part: "') + graph + pynutil.insert('"')
        final_graph = pynutil.add_weight(final_graph, -0.1)
        self.fst = self.add_tokens(final_graph)