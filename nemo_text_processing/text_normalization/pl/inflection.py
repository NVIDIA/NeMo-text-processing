# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels


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
    for fields in load_labels(get_abs_path(filepath)):
        abbreviation, word, grammar_file = fields
        if grammar_file.endswith(".tsv"):
            inflected = inflect_abbreviation(abbreviation, word, grammar_file)
        else:
            if len(word.split()) != len(grammar_file.split()):
                raise ValueError(f"Abbreviation and grammar fields must have matching word counts: {fields}")
            inflected = {"base": pynini.cross(abbreviation, word)}
        for slot, graph in inflected.items():
            graphs[slot] = graph if slot not in graphs else graphs[slot] | graph
    return {slot: graph.optimize() for slot, graph in graphs.items()}


def load_inflected_phrase_abbreviations(filepath: str, deterministic: bool = True) -> Dict[str, 'pynini.FstLike']:
    """Loads word-aligned abbreviation expansions with noun and adjective paradigms."""

    graphs = {}
    for fields in load_labels(get_abs_path(filepath)):
        abbreviation, phrase, grammars, noun_grammar = fields
        words = phrase.split()
        word_grammars = grammars.split()
        if len(words) != len(word_grammars):
            raise ValueError(f"Abbreviation and grammar fields must have matching word counts: {fields}")
        noun_indices = [
            index for index, grammar in enumerate(word_grammars) if grammar in {"ma", "mi", "mp", "nt", "f"}
        ]
        if len(noun_indices) != 1:
            raise ValueError(f"Expected one noun gender in abbreviation grammar: {fields}")
        noun_index = noun_indices[0]
        noun_gender = word_grammars[noun_index]
        noun_forms = inflect_noun(words[noun_index], noun_grammar)
        adjective_forms = []
        for word, grammar in zip(words, word_grammars):
            if grammar in {"ma", "mi", "mp", "nt", "f"}:
                adjective_forms.append(None)
            elif grammar == "i":
                adjective_forms.append(None)
            elif grammar == "adj":
                forms = adjective_inflection(word)
                from nemo_text_processing.text_normalization.pl.taggers.ordinal import complete_paradigm

                complete_paradigm(forms, complete=True)
                adjective_forms.append(forms)
            else:
                raise ValueError(f"Unknown abbreviation word grammar {grammar!r}: {fields}")
        if deterministic:
            graphs.setdefault("base", []).append(pynini.cross(abbreviation, phrase))
            continue
        for slot, noun in noun_forms.items():
            inflected_words = []
            for index, (word, forms) in enumerate(zip(words, adjective_forms)):
                if index == noun_index:
                    inflected_words.append(noun)
                elif forms is None:
                    inflected_words.append(word)
                else:
                    adjective_slot = f"{noun_gender}_{slot}" if slot.startswith("sg_") else slot
                    inflected_words.append(forms[adjective_slot])
            graph = pynini.cross(abbreviation, " ".join(inflected_words))
            graphs.setdefault(slot, []).append(graph)
    return {slot: pynini.union(*slot_graphs).optimize() for slot, slot_graphs in graphs.items()}


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
