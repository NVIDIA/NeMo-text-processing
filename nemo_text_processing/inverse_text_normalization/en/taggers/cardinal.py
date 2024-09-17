# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path, num_to_word
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        self.graph_two_digit = graph_teen | ((graph_ties) + delete_space + (graph_digit | pynutil.insert("0")))
        graph_hundred = pynini.cross("hundred", "")

        graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("0"))
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundred_as_thousand = pynini.union(graph_teen, graph_ties + delete_space + graph_digit)
        graph_hundred_as_thousand += delete_space + graph_hundred
        graph_hundred_as_thousand += delete_space + pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        graph_hundreds = graph_hundred_component | graph_hundred_as_thousand

        graph_ties_component = pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        )

        graph_ties_component_at_least_one_none_zero_digit = graph_ties_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_ties_component_at_least_one_none_zero_digit = graph_ties_component_at_least_one_none_zero_digit

        # %%% International numeric format
        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("thousand"),
            pynutil.insert("000", weight=0.1),
        )

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("million"),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("billion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("trillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("quadrillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("quintillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + self.delete_word("sextillion"),
            pynutil.insert("000", weight=0.1),
        )
        # %%%

        graph_int = (
            graph_sextillion
            + delete_space
            + graph_quintillion
            + delete_space
            + graph_quadrillion
            + delete_space
            + graph_trillion
            + delete_space
            + graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
        )

        # %% Indian numeric format simple https://en.wikipedia.org/wiki/Indian_numbering_system
        # This only covers "standard format".
        # Conventional format like thousand crores/lakh crores is yet to be implemented
        graph_in_thousands = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("thousand"),
            pynutil.insert("00", weight=0.1),
        )
        graph_in_lakhs = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("lakh") | pynutil.delete("lakhs")),
            pynutil.insert("00", weight=0.1),
        )

        graph_in_crores = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("crore") | pynutil.delete("crores")),
            pynutil.insert("00", weight=0.1),
        )

        graph_in_arabs = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("arab") | pynutil.delete("arabs")),
            pynutil.insert("00", weight=0.1),
        )

        graph_in_kharabs = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("kharab") | pynutil.delete("kharabs")),
            pynutil.insert("00", weight=0.1),
        )

        graph_in_nils = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("nil") | pynutil.delete("nils")),
            pynutil.insert("00", weight=0.1),
        )

        graph_in_padmas = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("padma") | pynutil.delete("padmas")),
            pynutil.insert("00", weight=0.1),
        )

        graph_in_shankhs = pynini.union(
            graph_ties_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("shankh") | pynutil.delete("shankhs")),
            pynutil.insert("00", weight=0.1),
        )

        graph_ind = (
            graph_in_shankhs
            + delete_space
            + graph_in_padmas
            + delete_space
            + graph_in_nils
            + delete_space
            + graph_in_kharabs
            + delete_space
            + graph_in_arabs
            + delete_space
            + graph_in_crores
            + delete_space
            + graph_in_lakhs
            + delete_space
            + graph_in_thousands
        )

        graph = pynini.union((graph_int | graph_ind) + delete_space + graph_hundreds, graph_zero,)

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT), "0"
        )

        labels_exception = [num_to_word(x) for x in range(0, 13)]

        if input_case == INPUT_CASED:
            labels_exception += [x.capitalize() for x in labels_exception]

        graph_exception = pynini.union(*labels_exception).optimize()

        graph = (
            pynini.cdrewrite(pynutil.delete("and"), NEMO_SPACE, NEMO_SPACE, NEMO_SIGMA)
            @ (NEMO_ALPHA + NEMO_SIGMA)
            @ graph
        ).optimize()

        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph)

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def delete_word(self, word: str):
        """Capitalizes word for `cased` input"""
        delete_graph = pynutil.delete(word).optimize()
        if self.input_case == INPUT_CASED:
            if len(word) > 0:
                delete_graph |= pynutil.delete(word[0].upper() + word[1:])

        return delete_graph.optimize()
