# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2022, 2023 Jim O'Regan for Språkbanken Tal
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
from typing import Dict, Iterable

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst, delete_space
from nemo_text_processing.text_normalization.pl.graph_utils import PL_ALPHA
from nemo_text_processing.text_normalization.pl.utils import adjective_inflection, get_abs_path, load_labels

CASES = ["nom", "gen", "dat", "acc", "ins", "loc", "voc"]
DEFAULT_SLOT = "mi_sg_nom"
SCALE_NAMES = ["tysiąc", "milion", "miliard", "bilion", "biliard", "trylion", "tryliard"]


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    exactly_three_digits = NEMO_DIGIT**3
    up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)
    cardinal_string = pynini.closure(NEMO_DIGIT, 1)
    cardinal_string |= (
        up_to_three_digits
        + pynutil.delete(" ")
        + pynini.closure(exactly_three_digits + pynutil.delete(" "))
        + exactly_three_digits
    )
    return cardinal_string @ fst


def get_digit_forms(filepath: str) -> Dict[str, Dict[str, object]]:
    output = {}
    for digit, grammar, form in load_labels(get_abs_path(filepath)):
        forms = output.setdefault(digit, {})
        if grammar not in forms:
            forms[grammar] = form
        elif isinstance(forms[grammar], list):
            forms[grammar].append(form)
        else:
            forms[grammar] = [forms[grammar], form]
    return output


def _forms_to_graphs(
    forms: Dict[str, Dict[str, object]], deterministic: bool
) -> Dict[str, Dict[str, 'pynini.FstLike']]:
    graphs = {}
    for number, slots in forms.items():
        graphs[number] = {}
        for slot, values in slots.items():
            values = values if isinstance(values, list) else [values]
            if deterministic:
                values = values[:1]
            graphs[number][slot] = pynini.union(*(pynini.cross(number, value) for value in values)).optimize()
    return graphs


def _invert_string_file(path: str) -> 'pynini.FstLike':
    return pynini.invert(pynini.string_file(get_abs_path(path))).optimize()


def _case_for_slot(slot: str) -> str:
    if slot == "compound":
        return slot
    for case in CASES:
        if slot == case or slot.endswith(f"_{case}"):
            return case
    raise ValueError(f"Cannot determine case from slot: {slot}")


def _select(mapping: Dict[str, 'pynini.FstLike'], keys: Iterable[str]) -> 'pynini.FstLike':
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of {list(keys)} is available")


def _noun_forms(lemma: str) -> Dict[str, str]:
    if lemma == "tysiąc":
        return {key: value for key, value in load_labels(get_abs_path("data/numbers/tysiac.tsv"))}

    stem = lemma
    loc_sg = "u"
    if lemma.endswith("ion"):
        loc_sg = "ie"
    elif lemma.endswith("iard"):
        loc_sg = "zie"
    return {
        "sg_nom": lemma,
        "sg_gen": lemma + "a",
        "sg_dat": lemma + "owi",
        "sg_acc": lemma,
        "sg_ins": lemma + "em",
        "sg_loc": stem + loc_sg,
        "sg_voc": lemma + "ie",
        "pl_nom": lemma + "y",
        "pl_gen": lemma + "ów",
        "pl_dat": lemma + "om",
        "pl_acc": lemma + "y",
        "pl_ins": lemma + "ami",
        "pl_loc": lemma + "ach",
        "pl_voc": lemma + "y",
    }


class CardinalFst(GraphFst):
    """Classifies Polish cardinal numbers and exposes each inflectional graph in ``graphs``."""

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit_forms = get_digit_forms("data/numbers/digit_forms.tsv")
        teen_forms = get_digit_forms("data/numbers/teens_forms.tsv")
        digit_graphs = _forms_to_graphs(digit_forms, deterministic)
        teen_graphs = _forms_to_graphs(teen_forms, deterministic)

        jeden = adjective_inflection("jeden", compound="jedno")
        from nemo_text_processing.text_normalization.pl.taggers.ordinal import complete_paradigm

        complete_paradigm(jeden, complete=True)
        self.jeden_all = {slot: pynini.cross("1", form) for slot, form in jeden.items()}

        zero_forms = {
            "sg_nom": "zero",
            "sg_gen": "zera",
            "sg_dat": "zeru",
            "sg_acc": "zero",
            "sg_ins": "zerem",
            "sg_loc": "zerze",
            "sg_voc": "zero",
        }
        self.zero_all = {slot: pynini.cross("0", form) for slot, form in zero_forms.items()}
        self.zero_sg = {slot[3:]: graph for slot, graph in self.zero_all.items()}

        ordinary_slots = set(self.jeden_all)
        for forms in digit_forms.values():
            ordinary_slots.update(forms)
        for forms in teen_forms.values():
            ordinary_slots.update(forms)

        tens_nom = _invert_string_file("data/numbers/tens.tsv")
        tens_gen = _invert_string_file("data/numbers/tens_gen.tsv")
        tens_ins = _invert_string_file("data/numbers/tens_ins.tsv")
        tens_compound = _invert_string_file("data/numbers/tens_prefix.tsv")
        hundreds_nom = _invert_string_file("data/numbers/hundreds.tsv")
        hundreds_gen = _invert_string_file("data/numbers/hundreds_gen.tsv")
        hundreds_ins = _invert_string_file("data/numbers/hundreds_ins.tsv")
        hundreds_compound = _invert_string_file("data/numbers/hundreds.tsv")

        join = pynutil.insert(" ")
        compound_join = pynutil.insert("")
        if not deterministic:
            compound_join |= pynutil.add_weight(pynutil.insert(" "), 0.001)

        self.graphs = {}
        self.two_digit_graphs = {}
        self.hundreds_graphs = {}

        for slot in sorted(ordinary_slots):
            case = _case_for_slot(slot)
            if case == "compound":
                tens = tens_compound
                hundreds = hundreds_compound
                component_join = compound_join
            elif case == "ins":
                tens = tens_ins
                hundreds = hundreds_ins
                component_join = join
            elif case in {"gen", "dat", "loc"} or slot.startswith("mp_"):
                tens = tens_gen
                hundreds = hundreds_gen
                component_join = join
            else:
                tens = tens_nom
                hundreds = hundreds_nom
                component_join = join

            digit = self._digit_for_slot(digit_graphs, slot)
            compound_digit = self._compound_digit_for_slot(digit_graphs, slot)
            teen = self._teen_for_slot(teen_graphs, slot)
            isolated_one = self._one_for_slot(slot, compound=False, deterministic=deterministic)
            compound_one = self._one_for_slot(slot, compound=True, deterministic=deterministic)
            two_digit = (
                teen
                | tens + pynutil.delete("0")
                | tens + component_join + (compound_digit | compound_one)
                | pynutil.delete("0") + (digit | isolated_one)
            ).optimize()
            hundred = (
                hundreds + pynutil.delete("00")
                | hundreds + component_join + two_digit
                | pynutil.delete("0") + two_digit
                | pynutil.delete("00") + (digit | isolated_one)
            ).optimize()

            self.two_digit_graphs[slot] = two_digit
            self.hundreds_graphs[slot] = hundred
        for slot, hundred in self.hundreds_graphs.items():
            self.graphs[slot] = self._make_full_number_graph(hundred, slot, deterministic)

        self.graph_dict = self.graphs
        compound_boundary = pynutil.delete("-")
        if not deterministic:
            compound_boundary += pynini.union(pynutil.insert(""), pynutil.add_weight(pynutil.insert(" "), 0.001))
        self.compound = (self.graphs["compound"] + compound_boundary + pynini.closure(PL_ALPHA, 1)).optimize()

        self.graph = filter_punctuation(self.graphs[DEFAULT_SLOT] | self.zero_all["sg_nom"]).optimize() | self.compound
        if not deterministic:
            self.graph = (
                filter_punctuation(pynini.union(*self.graphs.values(), *self.zero_all.values())).optimize()
                | self.compound
            )

        self.graph_unfiltered = self.graph
        optional_minus = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)
        final_graph = optional_minus + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        self.fst = self.add_tokens(final_graph).optimize()

    def _digit_for_slot(self, graphs, slot):
        choices = {
            "f_pl_nom": ["f_pl_nom", "mi_pl_nom"],
            "f_pl_ins": ["f_pl_ins", "pl_ins"],
            "mi_pl_ins": ["mi_pl_ins", "pl_ins"],
            "pl_ins": ["mi_pl_ins", "pl_ins"],
        }.get(slot, [slot])
        case = _case_for_slot(slot)
        if case in {"acc", "voc"}:
            choices += [slot.rsplit("_", 1)[0] + "_nom", "mi_pl_nom"]
        choices += [f"pl_{case}", "mi_pl_nom"]
        return pynini.union(*(_select(forms, choices) for forms in graphs.values())).optimize()

    def _compound_digit_for_slot(self, graphs, slot):
        if slot.startswith("mp_") and _case_for_slot(slot) in {"nom", "acc"}:
            choices = ["pl_gen", "mp_pl_nom"]
        elif slot == "pl_ins":
            choices = ["mi_pl_ins", "pl_ins"]
        else:
            choices = [slot]
        case = _case_for_slot(slot)
        choices += [f"pl_{case}", "mi_pl_nom"]
        return pynini.union(*(_select(forms, choices) for forms in graphs.values())).optimize()

    def _teen_for_slot(self, graphs, slot):
        case = _case_for_slot(slot)
        choices = [slot, f"pl_{case}"]
        if case in {"acc", "voc"}:
            choices += ["mp_pl_nom" if slot.startswith("mp_") else "mi_pl_nom"]
        choices += ["mi_pl_nom"]
        return pynini.union(*(_select(forms, choices) for forms in graphs.values())).optimize()

    def _one_for_slot(self, slot, compound, deterministic):
        if slot == "compound":
            return self.jeden_all[slot]
        if not compound:
            return self.jeden_all[slot]
        case = _case_for_slot(slot)
        graph = pynini.cross("1", "jeden")
        if not deterministic:
            key = slot if slot in self.jeden_all else f"mi_sg_{case}"
            graph |= pynutil.add_weight(self.jeden_all[key], 0.001)
        return graph.optimize()

    def _make_full_number_graph(self, group, slot, deterministic):
        case = _case_for_slot(slot)
        if case == "compound":
            short_input = pynini.closure(NEMO_DIGIT, 1, 3)
            pad = (
                short_input
                @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
                @ NEMO_DIGIT**3
            )
            return (pad @ group).optimize()

        scale_slot = {
            "nom": "mi_sg_nom",
            "acc": "mi_sg_acc",
            "voc": "mi_sg_voc",
            "gen": "pl_gen",
            "dat": "pl_dat",
            "ins": "mi_pl_ins",
            "loc": "pl_gen",
        }[case]
        scale_group = self.hundreds_graphs[scale_slot]

        plural_group = self._restrict_group(scale_group, "plural")
        quantity_group = self._restrict_group(scale_group, "quantity")
        non_one_group = self._restrict_group(scale_group, "non_one")
        factors = []
        for scale in reversed(SCALE_NAMES):
            forms = _noun_forms(scale)
            if case in {"nom", "acc", "voc"}:
                singular = forms[f"sg_{case}"]
                plural = forms[f"pl_{case}"]
                quantity = forms["pl_gen"]
                factor = (
                    pynutil.delete("000")
                    | pynini.cross("001", singular) + pynutil.insert(" ")
                    | plural_group + pynutil.insert(" " + plural + " ")
                    | quantity_group + pynutil.insert(" " + quantity + " ")
                )
            else:
                factor = (
                    pynutil.delete("000")
                    | pynini.cross("001", forms[f"sg_{case}"]) + pynutil.insert(" ")
                    | non_one_group + pynutil.insert(" " + forms[f"pl_{case}"] + " ")
                )
            if not deterministic:
                factor |= pynutil.add_weight(pynini.cross("001", "jeden " + forms[f"sg_{case}"] + " "), 0.001)
            factors.append(factor)

        padded = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT**24
        )
        full = None
        for factor in factors:
            full = factor if full is None else full + factor
        full += group | pynutil.delete("000")
        clean = pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
        return (padded @ full @ clean).optimize()

    @staticmethod
    def _restrict_group(group, kind):
        hundred = NEMO_DIGIT
        if kind == "plural":
            inputs = hundred + (NEMO_DIGIT - "1") + pynini.union("2", "3", "4")
        elif kind == "quantity":
            inputs = hundred + pynini.union(
                "1" + NEMO_DIGIT,
                (NEMO_DIGIT - "1") + pynini.union("0", "5", "6", "7", "8", "9"),
                (NEMO_DIGIT - "0") + "1",
            )
        else:
            inputs = NEMO_DIGIT**3 - "001" - "000"
        return inputs @ group
