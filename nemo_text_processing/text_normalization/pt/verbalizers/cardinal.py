# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.pt.graph_utils import NEMO_NOT_QUOTE, GraphFst, shift_cardinal_gender_pt


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing Portuguese cardinal numbers, e.g.
        cardinal { integer: "dois" } -> dois
        cardinal { integer: "dois" } -> duas (feminine context via shift_cardinal_gender_pt)
        cardinal { negative: "true" integer: "cinco" } -> menos cinco

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)
        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", "menos "), 0, 1)
        self.optional_sign = optional_sign

        integer = pynini.closure(NEMO_NOT_QUOTE, 1)
        self.integer = pynutil.delete(" \"") + integer + pynutil.delete("\"")

        integer = pynutil.delete("integer:") + self.integer

        # Generate masculine form (default)
        graph_masc = optional_sign + integer

        # Generate feminine form using Portuguese gender conversion
        graph_fem = shift_cardinal_gender_pt(graph_masc)

        self.graph_masc = pynini.optimize(graph_masc)
        self.graph_fem = pynini.optimize(graph_fem)

        # Default to masculine for standalone numbers
        # Context-aware gender selection will be handled by higher-level components
        graph = graph_masc

        if not deterministic:
            # For alternate renderings and contractions
            # Portuguese doesn't have apocope like Spanish, but may have contractions
            pass

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
