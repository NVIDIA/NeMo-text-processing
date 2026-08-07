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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from nemo_text_processing.text_normalization.pl.inflection import inflect_noun
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels
from pynini.lib import pynutil


def _case(slot: str) -> str:
    for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
        if slot == case or slot.endswith(f"_{case}"):
            return case
    raise ValueError(f"Cannot determine case from {slot!r}")


class MeasureFst(GraphFst):
    """Classifies integer measures with case-inflected masculine units."""

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        unit_graphs = {}
        for symbol, lemma, grammar_file in load_labels(get_abs_path("data/measures/units.tsv")):
            for slot, form in inflect_noun(lemma, grammar_file).items():
                graph = pynini.cross(symbol, form)
                unit_graphs[slot] = graph if slot not in unit_graphs else unit_graphs[slot] | graph
        unit_graphs = {slot: graph.optimize() for slot, graph in unit_graphs.items()}

        positive = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        one = pynini.accep("1")
        few = pynini.intersect(positive, pynini.closure(NEMO_DIGIT) + pynini.union("2", "3", "4"))
        few = pynini.difference(few, pynini.closure(NEMO_DIGIT) + pynini.union("12", "13", "14"))
        many = pynini.union("0", pynini.difference(pynini.difference(positive, one), few)).optimize()
        optional_space = pynini.closure(delete_space, 0, 1)

        def graph_for(number_input, number_graph, unit_slot):
            return (
                pynutil.insert('cardinal { integer: "')
                + (number_input @ number_graph)
                + pynutil.insert('" } units: "')
                + optional_space
                + unit_graphs[unit_slot]
                + pynutil.insert('"')
            )

        self.graphs = {}
        for slot, number_graph in cardinal.graphs.items():
            if slot == "compound":
                continue
            case = _case(slot)
            graph = graph_for(one, number_graph, f"sg_{case}")
            graph |= graph_for(few, number_graph, f"pl_{case}")
            graph |= graph_for(many, number_graph, "pl_gen")
            self.graphs[slot] = graph.optimize()

        self.graph_dict = self.graphs
        graph = self.graphs["mi_sg_nom"] if deterministic else pynini.union(*self.graphs.values())
        self.final_graph = graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
