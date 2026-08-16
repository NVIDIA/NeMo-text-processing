# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    NEMO_ALPHA_HE,
    GraphFst,
    delete_and,
    delete_optional_and,
)
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals in Hebrew
        e.g. מינוס עשרים ושלוש ("minus twenty three" in Hebrew)-> cardinal { negative: "-" integer: "23" } }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # digits
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        prefix_graph = pynini.string_file(get_abs_path("data/prefix.tsv"))

        # teens
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_ties += pynini.union(
            delete_space + delete_optional_and + graph_digit,
            pynutil.insert("0", weight=0.001),
        )
        graph_two_digit = pynini.union(graph_teen, graph_ties)

        self.graph_two_digit = pynini.union(graph_digit, graph_ties, pynutil.add_weight(graph_teen, -0.001))

        # hundreds
        hundred_exception = pynini.string_file(get_abs_path("data/numbers/hundreds_exception.tsv"))
        delete_hundred = pynutil.delete("מאות")
        graph_hundred = delete_optional_and + pynini.union(
            hundred_exception,
            graph_digit + delete_space + delete_hundred,
            pynutil.insert("0", weight=0.001),
        )
        graph_hundred += delete_space
        graph_hundred += pynini.union(
            delete_optional_and + graph_two_digit,
            pynutil.insert("0") + delete_space + delete_and + graph_digit,
            pynutil.insert("00", weight=0.001),
        )
        graph_hundred = pynini.union(
            graph_hundred,
            pynutil.insert("0") + graph_two_digit,
            pynutil.insert("00") + graph_digit,
        )
        self.graph_hundred = graph_hundred @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        )

        # thousands
        thousand_exception = pynini.string_file(get_abs_path("data/numbers/thousands_exception.tsv"))
        thousand_digit = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))
        delete_thousand = pynutil.delete("אלפים") | pynutil.delete("אלף", weight=0.001)

        large_number_prefix = pynini.union(
            graph_hundred,
            pynutil.insert("0") + graph_two_digit,
            pynutil.insert("00") + thousand_digit,
        )
        many_thousands = large_number_prefix + delete_space + delete_thousand
        graph_thousands = delete_optional_and + pynini.union(
            (pynutil.insert("00") + thousand_exception),
            many_thousands,
            pynutil.insert("000", weight=0.001),
        )

        self.graph_thousands = pynini.union(graph_thousands + delete_space + graph_hundred, graph_zero)
        self.graph_thousands @= pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT),
            "0",
        )

        # millions
        million_exceptions = pynini.string_file(get_abs_path("data/numbers/millions_exception.tsv"))
        million_exceptions = pynutil.insert("00") + million_exceptions
        delete_millions = pynutil.delete("מיליונים") | pynutil.delete("מיליון", weight=0.001)
        many_millions = large_number_prefix + delete_space + delete_millions
        graph_millions = pynini.union(many_millions, million_exceptions, pynutil.insert("000", weight=0.001))

        graph = pynini.union(
            graph_millions + delete_space + graph_thousands + delete_space + graph_hundred,
            graph_zero,
        )
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT),
            "0",
        )

        labels_exception = load_labels(get_abs_path("data/numbers/digit.tsv"))
        labels_exception = list(set([x[0] for x in labels_exception] + ["אפס", "עשר", "עשרה"]))
        labels_exception += ["ו" + label for label in labels_exception]
        graph_exception = pynini.union(*labels_exception).optimize()
        graph = ((NEMO_ALPHA_HE + NEMO_SIGMA) @ graph).optimize()

        self.graph_no_exception = graph

        ### Token insertion
        minus_graph = pynutil.insert("negative: ") + pynini.cross("מינוס", '"-"') + NEMO_SPACE
        optional_minus_graph = pynini.closure(minus_graph, 0, 1)

        optional_prefix_graph = pynini.closure(
            pynutil.insert('morphosyntactic_features: "') + prefix_graph + pynutil.insert('"') + insert_space,
            0,
            1,
        )

        graph_wo_small_digits = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        cardinal_wo_viable_hours = load_labels(get_abs_path("data/numbers/viable_hours.tsv"))
        cardinal_wo_viable_hours = list(set([x[0] for x in cardinal_wo_viable_hours]))
        viable_hours_exception = pynini.union(*cardinal_wo_viable_hours).optimize()
        self.graph_wo_viable_hours = (pynini.project(graph, "input") - viable_hours_exception.arcsort()) @ graph

        small_number_with_minus = (
            insert_space + minus_graph + pynutil.insert('integer: "') + self.graph_no_exception + pynutil.insert('"')
        )

        big_number_with_optional_minus = (
            optional_minus_graph + pynutil.insert('integer: "') + graph_wo_small_digits + pynutil.insert('"')
        )

        graph = optional_prefix_graph + (small_number_with_minus | big_number_with_optional_minus)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
