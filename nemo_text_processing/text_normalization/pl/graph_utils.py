# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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
from pynini.lib import byte, pynutil

from nemo_text_processing.text_normalization.en.graph_utils import delete_space, insert_space

from .utils import get_abs_path, load_labels

_ALPHA_UPPER = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŻŹ"
_ALPHA_LOWER = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzżź"

TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(_ALPHA_UPPER, _ALPHA_LOWER)])
TO_UPPER = pynini.invert(TO_LOWER)

PL_LOWER = pynini.union(*_ALPHA_LOWER).optimize()
PL_UPPER = pynini.union(*_ALPHA_UPPER).optimize()
PL_ALPHA = pynini.union(PL_LOWER, PL_UPPER).optimize()
PL_ALNUM = pynini.union(byte.DIGIT, PL_ALPHA).optimize()

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")

ensure_space = pynini.cross(pynini.closure(delete_space, 0, 1), " ")


def roman_to_int(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Alters given fst to convert Roman integers (lower and upper cased) into Arabic numerals. Valid for values up to 1000.
    e.g.
        "V" -> "5"
        "i" -> "1"

    Args:
        fst: Any fst. Composes fst onto Roman conversion outputs.
    """

    def _load_roman(file: str):
        roman = load_labels(get_abs_path(file))
        roman_numerals = [(x, y) for x, y in roman] + [(x.upper(), y) for x, y in roman]
        return pynini.string_map(roman_numerals)

    digit = _load_roman("data/roman/digit.tsv")
    ties = _load_roman("data/roman/ties.tsv")
    hundreds = _load_roman("data/roman/hundreds.tsv")

    graph = (
        digit
        | ties + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        | (
            hundreds
            + (ties | pynutil.add_weight(pynutil.insert("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        )
    ).optimize()

    return graph @ fst


def all_to_graph(graph_dict, default=None, deterministic=False):
    """
    Converts a dictionary of graphs to a single graph.
    Polish has multiple cases, so this is useful for generating a single graph
    """
    if default is None:
        for default_key in ["mi_sg_nom", "sg_nom", "nom", "mi_pl_nom", "pl_nom"]:
            if default_key in graph_dict:
                break
    else:
        default_key = default
    output_graph = graph_dict[default_key]
    if not deterministic:
        for key in graph_dict:
            if key != default_key:
                output_graph |= graph_dict[key]
    return output_graph.optimize()


def dict_to_graph(input_dict: dict, deterministic: bool = True) -> dict:
    """
    Converts a nested dictionary of forms to a dict of pynini.FSTs.
    Example input:
        {'2': {'mi_pl_ins': ['form1', 'form2'], 'mi_sg_nom': 'form3'}}
    Output:
        {'2': {'mi_pl_ins': FST, 'mi_sg_nom': FST}}
    """
    graph_dict = {}
    for key, value in input_dict.items():
        graph_dict[key] = {}
        for subkey, subvalue in value.items():
            if isinstance(subvalue, list):
                graph = pynini.cross(key, subvalue[0])
                if not deterministic:
                    for alt in subvalue[1:]:
                        graph |= pynini.cross(key, alt)
            else:
                graph = pynini.cross(key, subvalue)
            graph_dict[key][subkey] = graph
    return graph_dict
