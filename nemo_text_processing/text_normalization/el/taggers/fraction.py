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

from nemo_text_processing.text_normalization.el.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions in Greek, e.g.
        1/2 -> fraction { numerator: "ένα" denominator: "δεύτερο" }
        3/4 -> fraction { numerator: "τρία" denominator: "τέταρτα" }

    The numerator is a neuter cardinal and the denominator a neuter ordinal, singular when the
    numerator is one ("ένα τρίτο") and plural otherwise ("δύο τρίτα").

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        if ordinal is None:
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)

        num_graph = cardinal.graph_no_tokens
        num_no_one = pynini.difference(pynini.project(num_graph, "input"), pynini.accep("1")) @ num_graph

        denom_sg = ordinal.graph_neuter_sg
        denom_pl = ordinal.graph_neuter_pl

        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        numerator_one = pynutil.insert("numerator: \"") + pynini.cross("1", "ένα") + pynutil.insert("\"")
        numerator_multi = pynutil.insert("numerator: \"") + num_no_one + pynutil.insert("\"")

        branch_one = numerator_one + pynutil.delete("/") + pynutil.insert(" denominator: \"") + denom_sg + pynutil.insert("\"")
        branch_multi = (
            numerator_multi + pynutil.delete("/") + pynutil.insert(" denominator: \"") + denom_pl + pynutil.insert("\"")
        )

        graph = branch_one | branch_multi
        final_graph = optional_negative + graph
        self.final_graph = final_graph
        self.fst = self.add_tokens(final_graph).optimize()
