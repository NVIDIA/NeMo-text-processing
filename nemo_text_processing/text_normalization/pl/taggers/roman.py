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
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space, delete_space, insert_space
from nemo_text_processing.text_normalization.pl.graph_utils import roman_to_int
from nemo_text_processing.text_normalization.pl.inflection import case_prepositions, inflect_noun
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels
from pynini.lib import pynutil


def _name_forms(name: str, grammar_files: str):
    components = name.split(" ")
    grammars = grammar_files.split(",")
    if len(components) != len(grammars):
        raise ValueError(f"{name!r} must have one grammar per component")
    paradigms = [inflect_noun(component, grammar) for component, grammar in zip(components, grammars)]
    slots = set.intersection(*(set(paradigm) for paradigm in paradigms))
    return {slot: " ".join(paradigm[slot] for paradigm in paradigms) for slot in slots}


class RomanFst(GraphFst):
    """Classifies Roman ordinals following curated ruler and papal names."""

    def __init__(self, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        prepositions = case_prepositions()
        self.graphs = {}
        for category, name, grammar_files in load_labels(get_abs_path("data/roman/names.tsv")):
            gender = "f" if category == "queen" else "mp"
            for noun_slot, surface_name in _name_forms(name, grammar_files).items():
                number, case = noun_slot.split("_", 1)
                ordinal_slot = f"{gender}_{number}_{case}"
                if ordinal_slot not in ordinal.graphs:
                    continue
                name_graph = pynini.accep(surface_name)
                if category == "pope" and noun_slot == "sg_nom":
                    title = pynini.union("Papież", "papież") + delete_space + insert_space
                    name_graph |= title + pynini.accep(surface_name)
                graph = (
                    name_graph
                    + delete_space
                    + insert_space
                    + roman_to_int(ordinal.graphs[ordinal_slot])
                )
                if case in prepositions:
                    graph |= prepositions[case] + graph
                self.graphs[ordinal_slot] = (
                    graph if ordinal_slot not in self.graphs else self.graphs[ordinal_slot] | graph
                )

        self.graph_dict = {slot: graph.optimize() for slot, graph in self.graphs.items()}
        graph = pynini.union(*self.graph_dict.values()).optimize()
        self.graph = graph
        final_graph = pynutil.insert('name: "') + convert_space(graph) + pynutil.insert('"')
        self.fst = final_graph.optimize()
