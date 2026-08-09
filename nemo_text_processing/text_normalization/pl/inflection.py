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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.pl.utils import adjective_inflection, get_abs_path, load_labels


def _load_endings(grammar_file: str) -> Dict[str, str]:
    return {
        slot: "" if ending == "<eps>" else ending
        for slot, ending in load_labels(get_abs_path(f"data/grammar/{grammar_file}"))
    }


def inflect_noun(word: str, grammar_file: str) -> Dict[str, str]:
    """Inflects a noun using a grammar file containing slot-to-ending mappings."""

    endings = _load_endings(grammar_file)
    lemma_ending = endings["sg_nom"]
    if not word.endswith(lemma_ending):
        raise ValueError(f"{word!r} must end in {lemma_ending!r} from {grammar_file}")
    stem = word[: -len(lemma_ending)] if lemma_ending else word
    return {slot: stem + ending for slot, ending in endings.items()}


def load_numeric_nouns(filepath: str, grammar_file: str, trailing_zeros: int = 0) -> Dict[str, 'pynini.FstLike']:
    """Expands digits followed by noun endings into inflected numeral nouns."""

    endings = _load_endings(grammar_file)
    lemma_ending = endings["sg_nom"]
    optional_hyphen = pynini.closure(pynutil.delete("-"), 0, 1)
    graphs = {}
    for lemma, number in load_labels(get_abs_path(filepath)):
        if not lemma.endswith(lemma_ending):
            raise ValueError(f"{lemma!r} must end in {lemma_ending!r} from {grammar_file}")
        stem = lemma[: -len(lemma_ending)] if lemma_ending else lemma
        number += "0" * trailing_zeros
        for slot, ending in endings.items():
            graph = pynini.cross(number, stem) + optional_hyphen + pynini.accep(ending)
            graphs[slot] = graph if slot not in graphs else graphs[slot] | graph
    return {slot: graph.optimize() for slot, graph in graphs.items()}


def case_prepositions() -> Dict[str, 'pynini.FstLike']:
    """Loads prepositions as case-indexed identity graphs."""

    graphs = {}
    for preposition, cases in load_labels(get_abs_path("data/grammar/prepositions.tsv")):
        graph = pynini.accep(preposition) + pynutil.delete(" ") + pynutil.insert(" ")
        for case in cases.split(","):
            graphs[case] = graph if case not in graphs else graphs[case] | graph
    return {case: graph.optimize() for case, graph in graphs.items()}


def inflect_abbreviation(abbreviation: str, word: str, grammar_file: str) -> Dict[str, 'pynini.FstLike']:
    """Creates abbreviation-to-word graphs for every slot in a nominal paradigm."""

    endings = _load_endings(grammar_file)
    lemma_ending = endings["sg_nom"]
    if not abbreviation.endswith(lemma_ending) or not word.endswith(lemma_ending):
        raise ValueError(f"{abbreviation!r} and {word!r} must share the {lemma_ending!r} ending from {grammar_file}")
    abbreviation_stem = abbreviation[: -len(lemma_ending)] if lemma_ending else abbreviation
    word_stem = word[: -len(lemma_ending)] if lemma_ending else word
    return {
        slot: pynini.cross(abbreviation_stem + ending, word_stem + ending).optimize()
        for slot, ending in endings.items()
    }


def expand_abbreviation(abbreviation: str, word: str, grammar_file: str) -> Dict[str, 'pynini.FstLike']:
    """Creates graphs from one ambiguous abbreviation to each singular word form."""

    endings = _load_endings(grammar_file)
    lemma_ending = endings["sg_nom"]
    if not word.endswith(lemma_ending):
        raise ValueError(f"{word!r} must end in {lemma_ending!r} from {grammar_file}")
    word_stem = word[: -len(lemma_ending)] if lemma_ending else word
    return {
        slot: pynini.cross(abbreviation, word_stem + ending).optimize()
        for slot, ending in endings.items()
        if slot.startswith("sg_")
    }


def load_inflected_abbreviations(filepath: str) -> Dict[str, 'pynini.FstLike']:
    """Loads abbreviation, lemma, and grammar triples into slot-indexed graphs."""

    graphs = {}
    for abbreviation, word, grammar_file in load_labels(get_abs_path(filepath)):
        for slot, graph in inflect_abbreviation(abbreviation, word, grammar_file).items():
            graphs[slot] = graph if slot not in graphs else graphs[slot] | graph
    return {slot: graph.optimize() for slot, graph in graphs.items()}


def load_ambiguous_abbreviations(filepath: str) -> Dict[str, 'pynini.FstLike']:
    """Loads abbreviation, lemma, and grammar triples as singular alternatives."""

    graphs = {}
    for abbreviation, word, grammar_file in load_labels(get_abs_path(filepath)):
        for slot, graph in expand_abbreviation(abbreviation, word, grammar_file).items():
            graphs[slot] = graph if slot not in graphs else graphs[slot] | graph
    return {slot: graph.optimize() for slot, graph in graphs.items()}


def load_adjective_abbreviations(filepath: str) -> Dict[str, 'pynini.FstLike']:
    """Loads abbreviations whose adjective component exposes every inflectional slot."""

    from nemo_text_processing.text_normalization.pl.taggers.ordinal import complete_paradigm

    graphs = {}
    for abbreviation, prefix, adjective in load_labels(get_abs_path(filepath)):
        forms = adjective_inflection(adjective)
        complete_paradigm(forms, complete=True)
        for slot, form in forms.items():
            spoken = f"{prefix} {form}" if prefix else form
            graph = pynini.cross(abbreviation, spoken)
            graphs[slot] = graph if slot not in graphs else graphs[slot] | graph
    return {slot: graph.optimize() for slot, graph in graphs.items()}
