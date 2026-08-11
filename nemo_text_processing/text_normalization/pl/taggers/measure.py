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
from nemo_text_processing.text_normalization.pl.inflection import inflect_noun
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels


def _case(slot: str) -> str:
    for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
        if slot == case or slot.endswith(f"_{case}"):
            return case
    raise ValueError(f"Cannot determine case from {slot!r}")


class MeasureFst(GraphFst):
    """Classifies integer measures with case-inflected masculine units."""

    def __init__(
        self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        unit_graphs = {}
        unit_genders = set()
        for symbol, lemma, gender, grammar_file in load_labels(get_abs_path("data/measures/units.tsv")):
            unit_genders.add(gender)
            for slot, form in inflect_noun(lemma, grammar_file).items():
                graph = pynini.cross(symbol, form)
                key = (gender, slot)
                unit_graphs[key] = graph if key not in unit_graphs else unit_graphs[key] | graph
        unit_graphs = {slot: graph.optimize() for slot, graph in unit_graphs.items()}

        positive = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        one = pynini.accep("1")
        few = pynini.intersect(positive, pynini.closure(NEMO_DIGIT) + pynini.union("2", "3", "4"))
        few = pynini.difference(few, pynini.closure(NEMO_DIGIT) + pynini.union("12", "13", "14"))
        many = pynini.union("0", pynini.difference(pynini.difference(positive, one), few)).optimize()
        optional_space = pynini.closure(delete_space, 0, 1)

        def graph_for(number_input, number_graph, gender, unit_slot):
            return (
                pynutil.insert('cardinal { integer: "')
                + (number_input @ number_graph)
                + pynutil.insert('" } units: "')
                + optional_space
                + unit_graphs[(gender, unit_slot)]
                + pynutil.insert('"')
            )

        self.graphs = {}
        for gender in unit_genders:
            for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
                slot = f"{gender}_sg_{case}"
                one_graph = cardinal.graphs[slot]
                plural_graph = cardinal.graphs[f"{gender}_pl_{case}"]
                governed_slot = "pl_gen" if case in {"nom", "acc", "voc"} else f"pl_{case}"
                graph = graph_for(one, one_graph, gender, f"sg_{case}")
                graph |= graph_for(few, plural_graph, gender, f"pl_{case}")
                graph |= graph_for(many, plural_graph, gender, governed_slot)
                self.graphs[slot] = graph.optimize()

        self.graph_dict = self.graphs
        nominative = pynini.union(*(self.graphs[f"{gender}_sg_nom"] for gender in unit_genders))
        fractional_units = pynini.union(*(unit_graphs[(gender, "sg_gen")] for gender in unit_genders))

        def fractional_measure(graph, name, units=fractional_units):
            return (
                pynutil.insert(f"{name} {{ ")
                + graph
                + pynutil.insert(' } units: "')
                + optional_space
                + units
                + pynutil.insert('"')
            )

        self.decimal_graphs = {}
        for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
            decimal_graph = decimal.graphs[case]
            if not deterministic:
                decimal_graph |= decimal.digit_graphs[case]
            self.decimal_graphs[case] = fractional_measure(decimal_graph, "decimal").optimize()

        self.fraction_graphs = {}
        for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
            self.fraction_graphs[case] = fractional_measure(fraction.graphs[case], "fraction").optimize()

        self.fraction_graph = pynini.union(*self.fraction_graphs.values()).optimize()
        lexical_fraction_graph = pynini.Fst()
        if not deterministic:
            for gender in unit_genders:
                lexical = fraction.lexical_graphs.get(gender, pynini.Fst())
                if "all" in fraction.lexical_graphs:
                    lexical |= fraction.lexical_graphs["all"]
                lexical = pynutil.insert('value: "') + lexical + pynutil.insert('"')
                lexical_fraction_graph |= fractional_measure(
                    lexical, "fraction", unit_graphs[(gender, "sg_gen")]
                )
            lexical_fraction_graph = lexical_fraction_graph.optimize()
        graph = (
            nominative | self.decimal_graphs["nom"] | self.fraction_graphs["nom"]
            if deterministic
            else pynini.union(
                *self.graphs.values(),
                *self.decimal_graphs.values(),
                self.fraction_graph,
                lexical_fraction_graph,
            )
        )
        self.final_graph = graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
