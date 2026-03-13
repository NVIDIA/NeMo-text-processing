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

from nemo_text_processing.text_normalization.pt.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese ordinals, e.g.
        "1º" / "1ª" -> ordinal { integer: "primeiro" / "primeira" morphosyntactic_features: "gender_masc" / "gender_fem" }
        "21º" -> ordinal { integer: "vigésimo primeiro" morphosyntactic_features: "gender_masc" }

    Args:
        cardinal: CardinalFst instance for composing compound ordinals.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        spec_rows = load_labels(get_abs_path("data/ordinals/specials.tsv"))
        spec = {r[0]: r[1] for r in spec_rows if len(r) >= 2}
        conn_in = spec.get("connector_in", " e ")
        conn_out = spec.get("connector_out", " ")
        conn = pynini.cross(conn_in, conn_out)

        # Data: ordinal \t cardinal → FST cardinal→ordinal via load_labels
        digit_rows = load_labels(get_abs_path("data/ordinals/digit.tsv"))
        graph_digit = pynini.string_map([(r[1], r[0]) for r in digit_rows if len(r) >= 2]).optimize()
        teen_rows = load_labels(get_abs_path("data/ordinals/teen.tsv"))
        graph_teens = pynini.string_map([(r[1], r[0]) for r in teen_rows if len(r) >= 2]).optimize()
        ties_rows = load_labels(get_abs_path("data/ordinals/ties.tsv"))
        graph_ties = pynini.string_map([(r[1], r[0]) for r in ties_rows if len(r) >= 2]).optimize()
        hundreds_rows = load_labels(get_abs_path("data/ordinals/hundreds.tsv"))
        graph_hundreds = pynini.string_map([(r[1], r[0]) for r in hundreds_rows if len(r) >= 2]).optimize()

        graph_tens = pynini.union(
            graph_teens,
            graph_ties + pynini.closure(conn + graph_digit, 0, 1),
        )
        graph_hundred_component = pynini.union(
            graph_hundreds + pynini.closure(conn + pynini.union(graph_tens, graph_digit), 0, 1),
            graph_tens,
            graph_digit,
        )
        ordinal_rewrite = graph_hundred_component.optimize()
        ordinal_inner = cardinal_graph @ ordinal_rewrite

        opt_dot = pynini.closure(pynutil.delete("."), 0, 1)
        suffix_masc = opt_dot + pynutil.delete(pynini.union("º", "°"))
        suffix_fem = opt_dot + pynutil.delete("ª")
        digit_block = pynini.closure(NEMO_DIGIT, 1, 3)

        to_ordinal_masc = (digit_block + suffix_masc) @ ordinal_inner
        to_ordinal_fem = (digit_block + suffix_fem) @ ordinal_inner

        graph_masc = (
            pynutil.insert('integer: "')
            + to_ordinal_masc
            + pynutil.insert('" morphosyntactic_features: "gender_masc"')
        )
        graph_fem = (
            pynutil.insert('integer: "') + to_ordinal_fem + pynutil.insert('" morphosyntactic_features: "gender_fem"')
        )
        self.fst = self.add_tokens(pynini.union(graph_masc, graph_fem)).optimize()
