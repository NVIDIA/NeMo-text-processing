# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals, e.g.
        १०वां -> ordinal { integer: "दसवां" }
        २१वीं -> ordinal { integer: "इक्कीसवीं" }
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Load cardinal for number conversion
        cardinal = CardinalFst(deterministic=deterministic)
        
        # Define Hindi digits
        hindi_digits = pynini.union("०", "१", "२", "३", "४", "५", "६", "७", "८", "९")
        
        # Define acceptable input ordinal suffix variants and canonicalize outputs
        input_suffixes = pynini.union("वां", "वीं", "वें", "वे", "वी")
        suffix_canonical_map = pynini.string_map(
            [
                ("वां", "वां"),
                ("वीं", "वीं"),
                ("वी", "वीं"),
                ("वें", "वें"),
                ("वे", "वें"),
            ]
        )
        
        # Build ordinal pattern: one or more Hindi digits + accepted ordinal suffix variant
        ordinal_numbers = pynini.closure(hindi_digits, 1) + input_suffixes
        
        # For ordinal conversion, we need to:
        # 1. Extract the number part (without suffix)
        # 2. Convert number to cardinal form using existing cardinal FST
        # 3. Add the ordinal suffix back
        
        # Extract number part (remove suffix)
        number_part = pynini.closure(hindi_digits, 1)
        
        # Create conversion: Hindi number + suffix -> cardinal + suffix
        # Compose input (digits + suffix variant) with output (cardinal words + canonical suffix)
        input_acceptor = number_part + input_suffixes
        output_transducer = pynini.compose(number_part, cardinal.final_graph) + suffix_canonical_map
        ordinal_graph = pynini.compose(input_acceptor, output_transducer)
        
        # Final graph with token wrapping
        final_graph = pynutil.insert("integer: \"") + ordinal_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        
        self.fst = final_graph.optimize()
