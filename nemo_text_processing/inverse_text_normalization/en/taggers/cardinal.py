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
from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path, num_to_word
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted. 
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

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

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("thousand"),
            pynutil.insert("000", weight=0.1),
        )

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("million"),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("billion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("trillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("quadrillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("quintillion"),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("sextillion"),
            pynutil.insert("000", weight=0.1),
        )

        graph = pynini.union(
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
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT), "0"
        )

        labels_exception = [num_to_word(x) for x in range(0, 13)]
        graph_exception = pynini.union(*labels_exception)

        graph = (
            pynini.cdrewrite(pynutil.delete("and"), NEMO_SPACE, NEMO_SPACE, NEMO_SIGMA)
            @ (NEMO_ALPHA + NEMO_SIGMA)
            @ graph
        )

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
