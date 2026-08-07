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
from typing import Dict

import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst, insert_space
from nemo_text_processing.text_normalization.pl.graph_utils import all_to_graph
from nemo_text_processing.text_normalization.pl.utils import adjective_inflection, get_abs_path, load_labels
from pynini.lib import pynutil


def complete_paradigm(partial: Dict[str, str], complete: bool = False):
    partial["mi_sg_acc"] = partial["mi_sg_nom"]
    partial["mi_sg_loc"] = partial["mi_sg_ins"]
    partial["mi_sg_voc"] = partial["mi_sg_nom"]
    partial["ma_sg_nom"] = partial["mi_sg_nom"]
    partial["ma_sg_gen"] = partial["mi_sg_gen"]
    partial["ma_sg_dat"] = partial["mi_sg_dat"]
    partial["ma_sg_acc"] = partial["mi_sg_gen"]
    partial["ma_sg_ins"] = partial["mi_sg_ins"]
    partial["ma_sg_loc"] = partial["mi_sg_loc"]
    partial["ma_sg_voc"] = partial["mi_sg_voc"]
    for case in ["nom", "gen", "dat", "acc", "ins", "loc", "voc"]:
        partial[f"mp_sg_{case}"] = partial[f"ma_sg_{case}"]
    partial["nt_sg_gen"] = partial["mi_sg_gen"]
    partial["nt_sg_dat"] = partial["mi_sg_dat"]
    partial["nt_sg_acc"] = partial["nt_sg_nom"]
    partial["nt_sg_ins"] = partial["mi_sg_ins"]
    partial["nt_sg_loc"] = partial["mi_sg_loc"]
    partial["nt_sg_voc"] = partial["nt_sg_nom"]
    partial["f_sg_dat"] = partial["f_sg_gen"]
    partial["f_sg_acc"] = partial["f_sg_ins"]
    partial["f_sg_loc"] = partial["f_sg_gen"]
    partial["f_sg_voc"] = partial["f_sg_nom"]
    partial["mp_pl_acc"] = partial["pl_loc"]
    partial["mp_pl_voc"] = partial["mp_pl_nom"]
    partial["pl_nom"] = partial["nt_sg_nom"]
    partial["pl_gen"] = partial["pl_loc"]
    partial["pl_dat"] = partial["mi_sg_ins"]
    partial["pl_acc"] = partial["pl_nom"]
    partial["pl_voc"] = partial["pl_nom"]
    if complete:
        for gender in ["mi", "ma", "mp", "nt", "f"]:
            for case in ["nom", "gen", "dat", "acc", "ins", "loc", "voc"]:
                key = f"{gender}_pl_{case}"
                if key not in partial:
                    partial[key] = partial[f"pl_{case}"]


def make_graph_dict(filepath: str, invert: bool = True, complete: bool = False):
    output = {}
    for word, target in load_labels(get_abs_path(filepath)):
        forms = adjective_inflection(word)
        if complete:
            complete_paradigm(forms, complete=True)
        for slot, form in forms.items():
            source, destination = (target, form) if invert else (form, target)
            graph = pynini.cross(source, destination)
            output[slot] = graph if slot not in output else output[slot] | graph
    return {slot: graph.optimize() for slot, graph in output.items()}


class OrdinalFst(GraphFst):
    """Classifies Polish ordinals and exposes adjective-inflected graphs in ``graphs``."""

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        digits = make_graph_dict("data/ordinal/digit.tsv", complete=True)
        tens = make_graph_dict("data/ordinal/tens.tsv", complete=True)
        teens = make_graph_dict("data/ordinal/teens.tsv", complete=True)
        hundreds = make_graph_dict("data/ordinal/hundreds.tsv", complete=True)
        cardinal_hundreds = pynini.invert(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))).optimize()

        self.graphs = {}
        for slot in digits:
            joiner = pynutil.insert("") if slot == "compound" else insert_space
            if slot == "compound" and not deterministic:
                joiner |= pynutil.add_weight(insert_space, 0.001)

            two_digit = (
                tens[slot] + pynutil.delete("0")
                | pynutil.delete("0") + digits[slot]
                | teens[slot]
                | tens[slot] + joiner + digits[slot]
            ).optimize()
            three_digit = (
                hundreds[slot] + pynutil.delete("00")
                | pynutil.delete("0") + two_digit
                | cardinal_hundreds + joiner + two_digit
            ).optimize()
            short_input = pynini.closure(NEMO_DIGIT, 1, 3)
            pad = short_input @ pynini.cdrewrite(
                pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA
            ) @ NEMO_DIGIT**3
            self.graphs[slot] = (pad @ three_digit).optimize()

        self.graph_dict = self.graphs
        graph = all_to_graph(self.graphs, deterministic=deterministic)
        if not deterministic:
            graph = pynini.union(*self.graphs.values()).optimize()
        self.graph = (graph + pynutil.delete(".")).optimize()
        final_graph = pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        self.fst = self.add_tokens(final_graph).optimize()
