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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Hindi ordinals, e.g.
        १०वां -> ordinal { integer: "दसवां" }
        २१वीं -> ordinal { integer: "इक्कीसवीं" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        suffixes_list = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        suffixes_map = pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv"))
        suffixes_fst = pynini.union(suffixes_list, suffixes_map)
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))

        # Create English to Hindi digit mapping for preprocessing ordinals
        en_to_hi_digits = pynini.string_map([
            ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
            ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९")
        ])
        
        # Convert English digits to Hindi for ordinal processing
        # This handles cases like "1st" -> "१st" before ordinal matching
        digit_normalizer = pynini.cdrewrite(en_to_hi_digits, "", "", pynini.closure(NEMO_CHAR))

        graph = cardinal.final_graph + suffixes_fst
        exceptions = pynutil.add_weight(exceptions, -0.1)
        graph = pynini.union(exceptions, graph)
        
        # Apply digit normalization before ordinal matching
        # This allows both "1st" and "१st" to be processed as ordinals
        graph_with_normalization = pynini.compose(digit_normalizer, graph)

        # Store the core graph without token wrapping for use in other contexts (e.g., addresses)
        # For addresses, only expose exceptions to avoid over-matching (e.g., don't convert "210वीं" to "दो सौ दसवीं")
        exceptions_with_normalization = pynini.compose(digit_normalizer, exceptions)
        self.graph = graph_with_normalization.optimize()
        self.exceptions_graph = exceptions_with_normalization.optimize()
        
        final_graph = pynutil.insert("integer: \"") + graph_with_normalization + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
