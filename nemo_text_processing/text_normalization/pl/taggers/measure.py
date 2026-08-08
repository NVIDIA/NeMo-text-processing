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
from nemo_text_processing.text_normalization.pl.taggers.ordinal import complete_paradigm
from nemo_text_processing.text_normalization.pl.utils import adjective_inflection, get_abs_path, load_labels


def _case(slot: str) -> str:
    for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
        if slot == case or slot.endswith(f"_{case}"):
            return case
    raise ValueError(f"Cannot determine case from {slot!r}")


class MeasureFst(GraphFst):
    """Classifies integer measures with case-inflected masculine units."""

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
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
        integer_input = pynini.union("0", positive)
        digit_graph = pynini.string_map(
            [
                ("0", "zero"),
                ("1", "jeden"),
                ("2", "dwa"),
                ("3", "trzy"),
                ("4", "cztery"),
                ("5", "pięć"),
                ("6", "sześć"),
                ("7", "siedem"),
                ("8", "osiem"),
                ("9", "dziewięć"),
            ]
        )
        fractional_digits = digit_graph + pynini.closure(pynutil.insert(" ") + digit_graph)
        if not deterministic:
            fractional_digits |= positive @ cardinal.graphs["mi_sg_nom"]
        decimal_units = pynini.union(*(unit_graphs[(gender, "sg_gen")] for gender in unit_genders))
        self.decimal_graphs = {}
        for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
            integer_graph = cardinal.zero_all[f"sg_{case}"] | cardinal.graphs[f"mi_sg_{case}"]
            decimal = (
                pynutil.insert('decimal { integer_part: "')
                + (integer_input @ integer_graph)
                + pynutil.insert('" fractional_part: "')
                + pynutil.delete(pynini.union(",", "."))
                + fractional_digits
                + pynutil.insert('" } units: "')
                + optional_space
                + decimal_units
                + pynutil.insert('"')
            )
            self.decimal_graphs[case] = decimal.optimize()

        thousandth = adjective_inflection("tysięczny")
        complete_paradigm(thousandth, complete=True)
        self.fraction_graphs = {}
        for case in ("nom", "gen", "dat", "acc", "ins", "loc", "voc"):
            governed_case = "gen" if case in {"nom", "acc", "voc"} else case
            whole = integer_input @ (cardinal.zero_all[f"sg_{case}"] | cardinal.graphs[f"mi_sg_{case}"])
            case_graphs = []
            for width, denominator in ((1, "10"), (2, "100"), (3, "1000")):
                fraction_one = pynini.cross(f"{1:0{width}d}", "1")
                few_values = [
                    (f"{number:0{width}d}", str(number))
                    for number in range(2, 10**width)
                    if number % 10 in {2, 3, 4} and number % 100 not in {12, 13, 14}
                ]
                many_values = [
                    (f"{number:0{width}d}", str(number))
                    for number in range(2, 10**width)
                    if not (number % 10 in {2, 3, 4} and number % 100 not in {12, 13, 14})
                ]
                fraction_few = pynini.string_map(few_values)
                fraction_many = pynini.string_map(many_values)
                if denominator == "1000":
                    denominator_singular = pynutil.insert(thousandth[f"f_sg_{case}"])
                    denominator_few = pynutil.insert(thousandth[f"f_pl_{case}"])
                    denominator_plural = pynutil.insert(thousandth[f"f_pl_{governed_case}"])
                else:
                    denominator_singular = pynutil.insert(denominator) @ ordinal.graphs[f"f_sg_{case}"]
                    denominator_few = pynutil.insert(denominator) @ ordinal.graphs[f"f_pl_{case}"]
                    denominator_plural = pynutil.insert(denominator) @ ordinal.graphs[f"f_pl_{governed_case}"]
                fraction = (
                    (fraction_one @ cardinal.graphs[f"f_sg_{case}"])
                    + pynutil.insert(" ")
                    + denominator_singular
                )
                fraction |= (
                    (fraction_few @ cardinal.graphs[f"f_pl_{case}"])
                    + pynutil.insert(" ")
                    + denominator_few
                )
                fraction |= (
                    (fraction_many @ cardinal.graphs[f"f_pl_{case}"])
                    + pynutil.insert(" ")
                    + denominator_plural
                )
                case_graphs.append(
                    pynutil.insert('cardinal { integer: "')
                    + whole
                    + pynutil.insert(" i ")
                    + pynutil.delete(pynini.union(",", "."))
                    + fraction
                    + pynutil.insert('" } units: "')
                    + optional_space
                    + decimal_units
                    + pynutil.insert('"')
                )
            self.fraction_graphs[case] = pynini.union(*case_graphs).optimize()

        self.fraction_graph = pynini.union(*self.fraction_graphs.values()).optimize()
        graph = (
            nominative | self.fraction_graphs["nom"]
            if deterministic
            else pynini.union(*self.graphs.values(), *self.decimal_graphs.values(), self.fraction_graph)
        )
        self.final_graph = graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
