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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels
# from nemo_text_processing.text_normalization.pl.taggers.cardinal import cardinal_graph
from pynini.lib import pynutil


def adjective_inflection(word: str, compound: str = "") -> dict:
    def fill_bare_template(stem, mi_sg, mp_pl, vowel, stem_b="", compound=""):
        if stem_b == "":
            stem_b = stem
        if compound == "":
            compound = stem_b + "o"
        return {
            "mi_sg_nom": mi_sg,
            "mi_sg_gen": stem + "ego",
            "mi_sg_dat": stem + "emu",
            "mi_sg_ins": stem + vowel + "m",
            "nt_sg_nom": stem + "e",
            "f_sg_nom": stem_b + "a",
            "f_sg_gen": stem + "ej",
            "f_sg_ins": stem_b + "ą",
            "mp_pl_nom": mp_pl,
            "pl_ins": stem + vowel + "mi",
            "pl_loc": stem + vowel + "ch",
            "compound": compound,
        }
    stem_b = ""
    if word.endswith("en"):
        stem = word[:-2] + "n"
        mi_sg = word
        mp_pl = stem + "i"
        vowel = "y"
    elif word[-2:] in ["ni", "ci"]:
        stem = word
        mi_sg = word
        mp_pl = word
        vowel = ""
    elif word.endswith("szy"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word[:-2] + "i"
        vowel = "y"
    elif word.endswith("gi"):
        stem = word
        stem_b = word[:-1]
        mi_sg = word
        mp_pl = word[:-2] + "dzy"
        vowel = ""
    elif word.endswith("sty"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word[:-3] + "ści"
        vowel = "y"
    elif word.endswith("ty"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word[:-2] + "ci"
        vowel = "y"
    elif word.endswith("y"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word[:-1] + "i"
        vowel = "y"
    return fill_bare_template(stem, mi_sg, mp_pl, vowel, stem_b, compound)


def complete_paradigm(partial, complete=False):
    partial["mi_sg_acc"] = partial["mi_sg_nom"]
    partial["mi_sg_loc"] = partial["mi_sg_ins"]
    partial["mi_sg_voc"] = partial["mi_sg_nom"]
    # ma.sg same as mi.sg, except acc = gen
    partial["ma_sg_nom"] = partial["mi_sg_nom"]
    partial["ma_sg_gen"] = partial["mi_sg_gen"]
    partial["ma_sg_dat"] = partial["mi_sg_dat"]
    partial["ma_sg_acc"] = partial["mi_sg_gen"]
    partial["ma_sg_ins"] = partial["mi_sg_ins"]
    partial["ma_sg_loc"] = partial["mi_sg_loc"]
    partial["ma_sg_voc"] = partial["mi_sg_voc"]
    # mp.sg same as ma.sg
    partial["mp_sg_nom"] = partial["ma_sg_nom"]
    partial["mp_sg_gen"] = partial["ma_sg_gen"]
    partial["mp_sg_dat"] = partial["ma_sg_dat"]
    partial["mp_sg_acc"] = partial["ma_sg_acc"]
    partial["mp_sg_ins"] = partial["ma_sg_ins"]
    partial["mp_sg_loc"] = partial["ma_sg_loc"]
    partial["mp_sg_voc"] = partial["ma_sg_voc"]
    # nt.sg same as mi.sg aside from nom/acc/voc
    partial["nt_sg_gen"] = partial["mi_sg_gen"]
    partial["nt_sg_dat"] = partial["mi_sg_dat"]
    partial["nt_sg_acc"] = partial["nt_sg_nom"]
    partial["nt_sg_ins"] = partial["mi_sg_ins"]
    partial["nt_sg_loc"] = partial["mi_sg_loc"]
    partial["nt_sg_voc"] = partial["nt_sg_nom"]
    # f.sg
    partial["f_sg_dat"] = partial["f_sg_gen"]
    partial["f_sg_acc"] = partial["f_sg_ins"]
    partial["f_sg_loc"] = partial["f_sg_gen"]
    partial["f_sg_voc"] = partial["f_sg_nom"]
    # plurals
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
                key = f'{gender}_pl_{case}'
                if key not in partial:
                    partial[key] = partial[f'pl_{case}']


def make_graph_dict(filepath, invert=True, complete=False):
    output_graph = {}
    word_tsv = load_labels(get_abs_path(filepath))
    for word, target in word_tsv:
        word_forms = adjective_inflection(word)
        if complete:
            complete_paradigm(word_forms, complete=True)
        for key in word_forms:
            if invert:
                a = target
                b = word_forms[key]
            else:
                a = word_forms[key]
                b = target
            if key not in output_graph:
                output_graph[key] = pynini.cross(a, b)
            else:
                output_graph[key] |= pynini.cross(a, b)
    return output_graph


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        "2." -> ordinal { integer: "drugi" } }
        "2-gi" -> ordinal { integer: "drugi" } }
        "123." -> ordinal { integer: "sto dwudziesty trzeci" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        self.digits_all = make_graph_dict("data/ordinal/digit.tsv")
        self.tens_all = make_graph_dict("data/ordinal/tens.tsv")
        self.teens_all = make_graph_dict("data/ordinal/teens.tsv")
        self.hundreds_all = make_graph_dict("data/ordinal/hundreds.tsv")
        two_digit_all = self.make_two_digit()

        self.graph = (
            (
                pynini.closure(NEMO_DIGIT | pynini.accep("."))
                + pynutil.delete(pynutil.add_weight(pynini.union(*endings), weight=0.0001) | pynini.accep("."))
            )
            @ cardinal_graph
        ).optimize()
        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
    
    def make_two_digit(self):
        two_digits = {}
        for key in self.digits_all:
            two_digits[key] = self.tens_all[key] + pynutil.delete('0')
            two_digits[key] |= pynutil.delete('0') + self.digits_all[key]
            two_digits[key] |= self.teens_all[key]
            if key != "compound":
                two_digits[key] |= self.tens_all[key] + insert_space + self.digits_all[key]
            else:
                two_digits[key] |= self.tens_all[key] + self.digits_all[key]
        return two_digits
