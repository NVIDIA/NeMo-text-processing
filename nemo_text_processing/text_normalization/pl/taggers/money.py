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


def _inflect_currency(lemma, grammar_file, gender):
    noun, *adjective = lemma.split()
    forms = inflect_noun(noun, grammar_file)
    if not adjective:
        return forms
    adjective_forms = adjective_inflection(" ".join(adjective))
    complete_paradigm(adjective_forms, complete=True)
    adjective_slot = lambda slot: f"{gender}_{slot}" if slot.startswith("sg_") else slot
    return {slot: f"{form} {adjective_forms[adjective_slot(slot)]}" for slot, form in forms.items()}


def _scale_forms(scale):
    if scale == "tysiąc":
        return {"sg_nom": "tysiąc", "pl_nom": "tysiące", "pl_gen": "tysięcy"}
    return {"sg_nom": scale, "pl_nom": f"{scale}y", "pl_gen": f"{scale}ów"}


class MoneyFst(GraphFst):
    """Classifies Polish currency amounts written with a symbol or currency code."""

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currencies = {}
        quantities = dict(load_labels(get_abs_path("data/currency/quantities.tsv")))
        for fields in load_labels(get_abs_path("data/currency/currencies.tsv")):
            marker, lemma, gender, grammar_file = fields[:4]
            forms = _inflect_currency(lemma, grammar_file, gender)
            minor_forms = None
            minor_gender = None
            if len(fields) == 7:
                minor_forms = _inflect_currency(fields[4], fields[6], fields[5])
                minor_gender = fields[5]
            currencies[marker] = forms, gender, minor_forms, minor_gender

        positive = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        one = pynini.accep("1")
        few = pynini.intersect(positive, pynini.closure(NEMO_DIGIT) + pynini.union("2", "3", "4"))
        few -= pynini.closure(NEMO_DIGIT) + pynini.union("12", "13", "14")
        many = pynini.union("0", pynini.difference(pynini.difference(positive, one), few)).optimize()
        minor_join = pynutil.insert(" i ") if deterministic else pynutil.insert(" ")

        def amount(number, graph, unit):
            return (
                pynutil.insert('integer: "')
                + (number @ graph)
                + pynutil.insert('" currency: "')
                + pynutil.insert(unit)
                + pynutil.insert('"')
            )

        def amount_with_minor(number, graph, unit, minor_graph):
            return (
                pynutil.insert('integer: "')
                + (number @ graph)
                + pynutil.insert(f'" currency: "{unit}')
                + minor_join
                + pynutil.delete(",")
                + minor_graph
                + pynutil.insert('"')
            )

        def currency_graph(marker, forms, gender, minor_forms, minor_gender):
            marker_graph = pynutil.delete(marker)
            optional_space = pynini.closure(delete_space, 0, 1)
            singular_slot = f"{gender}_sg_nom" if f"{gender}_sg_nom" in cardinal.graphs else "mi_sg_nom"
            plural_slot = f"{gender}_pl_nom" if f"{gender}_pl_nom" in cardinal.graphs else "mi_pl_nom"
            one_graph = amount(one, cardinal.graphs[singular_slot], forms["sg_nom"])
            few_graph = amount(few, cardinal.graphs[plural_slot], forms["pl_nom"])
            many_graph = amount(many, cardinal.graphs[plural_slot], forms["pl_gen"])
            amount_graph = one_graph | few_graph | many_graph
            integer = marker_graph + optional_space + amount_graph
            integer |= amount_graph + optional_space + marker_graph
            if minor_forms is not None:
                minor_values = pynini.union(
                    *(pynini.cross(f"{value:02}", str(value)) for value in range(1, 100))
                )
                minor_singular_slot = (
                    f"{minor_gender}_sg_nom" if f"{minor_gender}_sg_nom" in cardinal.graphs else "mi_sg_nom"
                )
                minor_plural_slot = (
                    f"{minor_gender}_pl_nom" if f"{minor_gender}_pl_nom" in cardinal.graphs else "mi_pl_nom"
                )
                minor_one = (
                    (one @ cardinal.graphs[minor_singular_slot])
                    + pynutil.insert(" ")
                    + pynutil.insert(minor_forms["sg_nom"])
                )
                minor_few = (
                    (few @ cardinal.graphs[minor_plural_slot])
                    + pynutil.insert(" ")
                    + pynutil.insert(minor_forms["pl_nom"])
                )
                minor_many = (
                    (many @ cardinal.graphs[minor_plural_slot])
                    + pynutil.insert(" ")
                    + pynutil.insert(minor_forms["pl_gen"])
                )
                minor = minor_values @ (minor_one | minor_few | minor_many)
                minor_integer = (
                    amount_with_minor(one, cardinal.graphs[singular_slot], forms["sg_nom"], minor)
                    | amount_with_minor(few, cardinal.graphs[plural_slot], forms["pl_nom"], minor)
                    | amount_with_minor(many, cardinal.graphs[plural_slot], forms["pl_gen"], minor)
                )
                integer |= marker_graph + optional_space + minor_integer
                integer |= minor_integer + optional_space + marker_graph
            decimal_graph = (
                pynutil.insert("decimal { ")
                + decimal.graphs["nom"]
                + pynutil.insert(' } currency: "')
                + forms["sg_gen"]
                + pynutil.insert('"')
            )
            fraction_graph = (
                pynutil.insert("fraction { ")
                + fraction.graphs["nom"]
                + pynutil.insert(' } currency: "')
                + forms["sg_gen"]
                + pynutil.insert('"')
            )
            decimal_or_fraction = decimal_graph | fraction_graph
            quantity_graphs = []
            for quantity, scale in quantities.items():
                scale_forms = _scale_forms(scale)
                quantity_spoken = (
                    pynutil.insert('integer: "')
                    + (one @ cardinal.graphs[singular_slot])
                    + pynutil.insert(f' {scale_forms["sg_nom"]}" currency: "')
                    + pynutil.insert(forms["pl_gen"])
                    + pynutil.insert('"')
                )
                quantity_spoken |= (
                    pynutil.insert('integer: "')
                    + (few @ cardinal.graphs[plural_slot])
                    + pynutil.insert(f' {scale_forms["pl_nom"]}" currency: "')
                    + pynutil.insert(forms["pl_gen"])
                    + pynutil.insert('"')
                )
                quantity_spoken |= (
                    pynutil.insert('integer: "')
                    + (many @ cardinal.graphs[plural_slot])
                    + pynutil.insert(f' {scale_forms["pl_gen"]}" currency: "')
                    + pynutil.insert(forms["pl_gen"])
                    + pynutil.insert('"')
                )
                quantity_graphs.append(
                    quantity_spoken
                    + pynutil.delete(" ")
                    + pynutil.delete(quantity)
                    + optional_space
                    + marker_graph
                )

            return (
                marker_graph + optional_space + decimal_or_fraction
                | decimal_or_fraction + optional_space + marker_graph
                | integer
                | pynini.union(*quantity_graphs)
            ).optimize()

        self.final_graph = pynini.union(
            *(
                currency_graph(marker, forms, gender, minor_forms, minor_gender)
                for marker, (forms, gender, minor_forms, minor_gender) in currencies.items()
            )
        ).optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
