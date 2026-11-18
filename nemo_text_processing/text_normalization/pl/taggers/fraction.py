# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from nemo_text_processing.text_normalization.pl.taggers.cardinal import CASES
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels


class FractionFst(GraphFst):
    """Classifies Polish slash, mixed, and Unicode fractions."""

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        positive = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        one = pynini.accep("1")
        few = pynini.intersect(positive, pynini.closure(NEMO_DIGIT) + pynini.union("2", "3", "4"))
        few = pynini.difference(few, pynini.closure(NEMO_DIGIT) + pynini.union("12", "13", "14"))
        many = pynini.difference(positive, one | few).optimize()
        slash = pynini.closure(delete_space, 0, 1) + pynutil.delete("/") + pynini.closure(delete_space, 0, 1)
        canonical = positive + pynini.accep("/") + positive
        canonical |= pynini.string_file(get_abs_path("data/numbers/fraction.tsv"))

        self.graphs = {}
        for case in CASES:
            governed_case = "gen" if case in {"nom", "acc", "voc"} else case
            fraction = (
                pynutil.insert('numerator: "')
                + (one @ cardinal.graphs[f"f_sg_{case}"])
                + pynutil.insert('" denominator: "')
                + slash
                + (positive @ ordinal.graphs[f"f_sg_{case}"])
                + pynutil.insert('"')
            )
            fraction |= (
                pynutil.insert('numerator: "')
                + (few @ cardinal.graphs[f"f_pl_{case}"])
                + pynutil.insert('" denominator: "')
                + slash
                + (positive @ ordinal.graphs[f"f_pl_{case}"])
                + pynutil.insert('"')
            )
            fraction |= (
                pynutil.insert('numerator: "')
                + (many @ cardinal.graphs[f"f_pl_{case}"])
                + pynutil.insert('" denominator: "')
                + slash
                + (positive @ ordinal.graphs[f"f_pl_{governed_case}"])
                + pynutil.insert('"')
            )
            fraction = (canonical @ fraction).optimize()
            integer = (
                pynutil.insert('integer_part: "')
                + (positive @ cardinal.graphs[f"mi_sg_{case}"])
                + pynutil.insert('" ')
                + delete_space
            )
            half_input = pynini.union("1/2", "½")
            mixed_half = integer + pynutil.delete(half_input) + pynutil.insert('value: "pół"')
            non_half = (pynini.project(fraction, "input") - half_input) @ fraction
            mixed = integer + (fraction if not deterministic else non_half)
            self.graphs[case] = (fraction | mixed | mixed_half).optimize()

        self.lexical_graphs = {}
        for written, spoken, gender in load_labels(get_abs_path("data/numbers/fraction_lexical_nondet.tsv")):
            lexical = pynini.cross(written, spoken)
            self.lexical_graphs[gender] = (
                lexical if gender not in self.lexical_graphs else self.lexical_graphs[gender] | lexical
            )
        self.lexical_graphs = {gender: graph.optimize() for gender, graph in self.lexical_graphs.items()}
        self.lexical_graph = pynini.union(*self.lexical_graphs.values()).optimize()

        graph = self.graphs["nom"]
        if not deterministic:
            lexical = pynutil.insert('value: "') + self.lexical_graph + pynutil.insert('"')
            graph = pynini.union(*self.graphs.values(), lexical).optimize()
        self.final_graph = graph
        self.fst = self.add_tokens(graph).optimize()
