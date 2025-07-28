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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst
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


def make_graph_dict(filepath):
    output_graph = {}

    word_tsv = load_labels(get_abs_path(filepath))
    for word, target in word_tsv:
        word_forms = adjective_inflection(word)

        for key in word_forms:
            if key in output_graph:
                output_graph[key] = pynini.cross(word_forms[key], target)
            else:
                output_graph[key] |= pynini.cross(word_forms[key], target)
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

        digits_tsv = load_labels(get_abs_path("data/ordinal/digit.tsv"))
        for digit in digits_tsv:
            word, num = digit
            word_forms = adjective_inflection(word)



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
