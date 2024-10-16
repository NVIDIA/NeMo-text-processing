# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "-4 1/3" ->
    tokens { fraction { negative: "true" integer_part: "quatre" numerator: "un" denominator: "trois" morphosyntactic_features: "ième" } }
    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction will be generated (used for audio-based normalization) - TBD
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinals = cardinal.all_nums_no_tokens
        sing_numerator = pynini.accep("1") @ cardinals
        pl_numerators = (pynini.closure(NEMO_DIGIT) - "1") @ cardinals

        add_denom_suffix = pynini.closure(NEMO_DIGIT) + pynutil.insert("e")
        denominators = add_denom_suffix @ ordinal.graph
        change_denom_label = pynini.cross("integer", "denominator")
        pluralize_denom = pynini.closure(NEMO_SIGMA) + pynini.cross("\"ième\"", "\"ièmes\"")

        sing_fraction_graph = (
            pynutil.insert("numerator: \"")
            + sing_numerator
            + pynutil.insert("\" ")
            + pynutil.delete("/")
            + (denominators @ (change_denom_label + pynini.closure(NEMO_SIGMA)))
        )

        pl_fraction_graph = (
            pynutil.insert("numerator: \"")
            + pl_numerators
            + pynutil.insert("\" ")
            + pynutil.delete("/")
            + (denominators @ (change_denom_label + pluralize_denom))
        )

        integer_part = pynutil.insert("integer_part: \"") + cardinals + pynutil.insert("\"")
        optional_integer_part = pynini.closure(integer_part + pynini.accep(" "), 0, 1)

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + optional_integer_part + (sing_fraction_graph | pl_fraction_graph)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
