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

import csv
import os


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file
        
    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


def load_labels(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns dictionary of mappings
    """
    with open(abs_path, encoding="utf-8") as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
        return labels


def adjective_inflection(word: str, compound: str = "") -> dict:
    """
    inflect adjectives based on their endings.
    This includes things like ordinals and 'jeden' (1) which inflect like adjectives.
    """
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
    elif word.endswith("ony"):
        stem = word[:-1]
        mi_sg = word
        mp_pl = word[:-3] + "eni"
        vowel = "y"
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
    elif word.endswith("ki"):
        stem = word
        stem_b = word[:-1]
        mi_sg = word
        mp_pl = word[:-2] + "cy"
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
