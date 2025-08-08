# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses, URLs, etc.
        e.g. c d f ett snabel-a a b c punkt e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
        e.g. w w w punkt nvidia punkt com -> tokens { electronic { protocol: "www." domain: "nvidia.com" } }

    Args:
        tn_electronic_tagger: TN electronic tagger
        tn_electronic_verbalizer: TN electronic verbalizer
    """

    def __init__(
        self,
        tn_electronic_tagger: GraphFst,
        tn_electronic_verbalizer: GraphFst,
        project_input: bool = False
    ):
        super().__init__(name="electronic", kind="classify", project_input=project_input)

        # Invert the TN electronic verbalizer to go from Swedish verbal form back to structured format
        # This should produce the same token structure as TN (both protocol and domain for URLs)
        verbalizer_inverted = pynini.invert(tn_electronic_verbalizer.graph).optimize()

        # Use add_tokens which will handle the projecting/non-projecting cases
        final_graph = self.add_tokens(verbalizer_inverted)
        self.fst = final_graph.optimize()
