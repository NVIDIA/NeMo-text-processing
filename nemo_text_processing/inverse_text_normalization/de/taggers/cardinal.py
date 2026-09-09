# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from collections import defaultdict
from typing import List

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    TO_LOWER,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinal numbers, e.g.
        minus eine billion fünfundsechzig milliarden vier millionen sechs -> cardinal { negative: "-" integer: "1.065.004.000.006" }
    The transducer implements a period separator every three digits by default.
    Numbers below thirteen are not converted.
    Allows both compound numeral strings or separated by whitespace.
    "und" (en: "and") can be inserted between "hundert" and following number or "tausend" and following single or double digit number.

        e.g. minus drei und zwanzig -> cardinal { negative: "-" integer: "23" }
        e.g. minus dreiundzwanzig -> cardinal { negative: "-" integer: "23" }
        e.g. dreizehn -> cardinal { integer: "13" }
        e.g. ein hundert -> cardinal { integer: "100" }
        e.g. einhundert -> cardinal { integer: "100" }
        e.g. ein tausend -> cardinal { integer: "1.000" }
        e.g. eintausend -> cardinal { integer: "1.000" }
        e.g. ein tausend zwanzig -> cardinal { integer: "1.020" }
        e.g. kapitel drei -> cardinal { morphosyntactic_features: "kapitel" integer: "3" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit_no_one = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_one = pynini.string_file(get_abs_path("data/numbers/ones.tsv"))
        graph_digit = graph_digit_no_one | graph_one
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_minus = pynini.string_file(get_abs_path("data/numbers/minus.tsv"))
        graph_quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
        and_words = [row[0] for row in load_labels(get_abs_path("data/numbers/und.tsv"))]

        # German speaks the ones before the tens, e.g. "einundzwanzig" is 21,
        # so the two digits cannot be read off in the order they are spoken
        graph_ties_digit = self.get_ties_digit(
            [get_abs_path("data/numbers/digit.tsv"), get_abs_path("data/numbers/ones.tsv")],
            get_abs_path("data/numbers/ties.tsv"),
            and_words,
        )
        delete_und = self.delete_word(pynini.union(*and_words)).ques

        self.graph_two_digit = (graph_teen | (graph_ties + pynutil.insert("0")) | graph_ties_digit).optimize()
        graph_two_digit = self.graph_two_digit

        # isolated subgraphs handed to the other semiotic classes
        self.digits = graph_digit.optimize()
        self.graph_double_digits = self.graph_two_digit
        self.graph_single_and_double_digits = (graph_digit | graph_two_digit).optimize()

        delete_hundert = self.delete_word(self.get_quantity(graph_quantities, "100"))
        multiplier = (graph_digit | pynutil.insert("1")) + delete_space
        graph_hundred_component = (
            (multiplier + delete_hundert + delete_space + delete_und + delete_space + graph_two_digit)
            | (
                multiplier
                + delete_hundert
                + pynutil.insert("0")
                + delete_space
                + delete_und
                + delete_space
                + graph_digit
            )
            | (multiplier + delete_hundert + pynutil.insert("00"))
        )

        # digits are grouped in clusters of three, written right to left and period separated
        graph_cluster = (
            graph_hundred_component
            | (pynutil.insert("0") + graph_two_digit)
            | (pynutil.insert("00") + graph_digit)
            | pynutil.insert("000")
        )
        graph_cluster_non_zero = (
            graph_hundred_component | (pynutil.insert("0") + graph_two_digit) | (pynutil.insert("00") + graph_digit)
        )

        non_zero_digits = pynini.difference(NEMO_DIGIT, "0")
        chars_to_remove = pynini.accep("0") | pynini.accep(".")
        remove_chars = pynutil.delete(pynini.closure(chars_to_remove))
        remove_leading_zeros = pynini.cdrewrite(remove_chars, "[BOS]", non_zero_digits, NEMO_SIGMA)
        remove_period_separators = pynini.cdrewrite(pynutil.delete("."), "", "", NEMO_SIGMA)

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_cluster_non_zero @ remove_leading_zeros
        ).optimize()

        def magnitude(written, lower_magnitudes, leading_cluster=graph_cluster, empty_multiplier=True):
            """
            WFST grammar for one order of magnitude, e.g. "million"

            Args:
                written: written form of the quantity, e.g. 1.000.000
                lower_magnitudes: WFST grammar for the next magnitude down, e.g. "thousands"
                leading_cluster: WFST grammar for the cluster multiplying the quantity
                empty_multiplier: whether the magnitude may be skipped, e.g. "eine million drei"
            """
            quantity = self.get_quantity(graph_quantities, written)
            delete_quantity = self.delete_word(quantity)
            multiplied = delete_quantity + pynutil.insert("1.") + delete_space + delete_und | (
                leading_cluster + delete_space + delete_quantity + pynutil.insert(".") + delete_und
            )
            if empty_multiplier:
                multiplied |= pynutil.insert("000.")
            standalone = graph_quantities @ pynini.accep(written)
            return standalone | (multiplied + delete_space + lower_magnitudes)

        magnitudes = self.get_magnitudes(get_abs_path("data/numbers/quantities.tsv"))
        graph_magnitudes = []
        lower_magnitudes = graph_cluster
        for written in magnitudes:
            lower_magnitudes = magnitude(written, lower_magnitudes)
            graph_magnitudes.append(lower_magnitudes)

        graph = pynini.union(*graph_magnitudes, graph_cluster, graph_zero)
        graph = graph @ remove_leading_zeros

        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph)
            graph_minus = capitalized_input_graph(graph_minus)

        # where a reading is ambiguous the more specific class wins, e.g. a decimal over 1.000.000
        # and a year over 2.020
        graph = pynutil.add_weight(graph, weight=0.001)

        # only a standalone cardinal is rendered with period separators; every class built on top of
        # this one embeds the plain digit string, so the two readings are kept apart by name
        graph_with_separators = graph.optimize()
        graph_without_separators = (graph @ remove_period_separators).optimize()

        self.graph_no_exception = graph_without_separators
        # alias under the name the standalone German grammars use
        self.graph_all_cardinals = graph_without_separators

        # years 0 - 9999, including the colloquial readings, e.g. "zwanzigvierundzwanzig" -> 2024
        first_millenium = graph_cluster_non_zero
        second_tenth_millenium = magnitude(
            magnitudes[0], graph_cluster, leading_cluster=graph_cluster_non_zero, empty_multiplier=False
        )
        ten = pynini.project(graph_teen @ pynini.accep("10"), "input")
        graph_11_99 = (pynini.project(graph_two_digit, "input") - ten) @ graph_two_digit

        # single digit year tails take a leading zero, e.g. "neunzehnhundertfünf" and
        # "neunzehnhundertnullfünf" both denormalize to 1905
        single_digit_years = (pynutil.insert("0") + graph_digit) | (graph_zero + delete_space + graph_digit)

        years_exceptions = (
            graph_11_99
            + delete_space
            + delete_hundert.ques
            + delete_space
            + (graph_two_digit | single_digit_years | pynutil.insert("00"))
        )
        years = first_millenium | second_tenth_millenium | years_exceptions
        self.graph_years = (years @ remove_leading_zeros @ remove_period_separators).optimize()

        # a bare cardinal from zero to twelve inclusive stays spelled out, e.g. "drei" -> "drei"
        spelled_out = graph_zero | graph_digit | (graph_teen @ pynini.union("10", "11", "12"))
        self.dozen = spelled_out.optimize()

        # the classes built on top of this one withhold only the single digits, so that an ordinal
        # such as "zehnter" still denormalizes to "10." while "zweiter" stays "zweiter"
        single_digits = graph_zero | graph_digit

        if input_case == INPUT_CASED:
            spelled_out = capitalized_input_graph(spelled_out)
            single_digits = capitalized_input_graph(single_digits)
        spelled_out = pynini.project(spelled_out, "input")
        single_digits = pynini.project(single_digits, "input")

        self.graph = (
            (pynini.project(graph_without_separators, "input") - single_digits.arcsort())
            @ graph_without_separators
        ).optimize()

        self.optional_minus_graph = pynini.closure(
            pynutil.insert('negative: "') + graph_minus + pynutil.delete(NEMO_SPACE) + pynutil.insert('" '), 0, 1
        )
        # alias under the name the standalone German grammars use
        self.optional_negative = self.optional_minus_graph

        # standalone readings, the only place the period separators are emitted

        # fully denormalized, used where the context rules out spelling numbers out
        self.forced_integer_graph_with_separators = (
            self.optional_minus_graph + pynutil.insert('integer: "') + graph_with_separators + pynutil.insert('"')
        ).optimize()

        # canonical, leaving the first dozen spelled out
        canonical_graph = spelled_out | (
            (pynini.project(graph_with_separators, "input") - spelled_out.arcsort()) @ graph_with_separators
        )
        self.canonical_integer_graph_with_separators = (
            self.optional_minus_graph + pynutil.insert('integer: "') + canonical_graph + pynutil.insert('"')
        ).optimize()

        # a noun such as "kapitel" in the determiner position forces the numeral to denormalize,
        # e.g. "kapitel drei" -> "kapitel 3" even though a bare "drei" stays spelled out
        nouns_forcing_denormalization = pynini.string_file(
            get_abs_path("data/numbers/nouns_forcing_denormalization.tsv")
        )
        if input_case == INPUT_CASED:
            nouns_forcing_denormalization = pynini.project(
                capitalized_input_graph(nouns_forcing_denormalization), "input"
            )
        graph_forced_denormalization = (
            pynutil.insert('morphosyntactic_features: "')
            + nouns_forcing_denormalization
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + self.forced_integer_graph_with_separators
        )

        final_graph = self.add_tokens(
            self.canonical_integer_graph_with_separators | graph_forced_denormalization
        )
        self.fst = final_graph.optimize()

    def delete_word(self, word: 'pynini.FstLike') -> 'pynini.FstLike':
        """
        Deletes an acceptor, also matching its capitalized form for `cased` input.
        German capitalizes nouns mid-sentence, so quantity words such as "Millionen" need this
        even when they are not sentence initial.

        Args:
            word: acceptor for the spoken form(s) to delete
        Returns:
            res: fst deleting the word
        """

        if self.input_case == INPUT_CASED:
            word |= pynini.project(pynini.compose(TO_LOWER + NEMO_SIGMA, word), "input")
        return pynutil.delete(word).optimize()

    def get_ties_digit(self, digit_paths: List[str], tie_path: str, and_words: List[str]) -> 'pynini.FstLike':
        """
        getting all denormalizations for numbers between 21 - 100

        Args:
            digit_paths: files to the digit tsvs, e.g. digit.tsv for 2 - 9 and ones.tsv for 1
            tie_path: file to tie tsv, e.g. 20, 30, etc.
            and_words: connectors between the digit and the tie, e.g. ["und"]
        Returns:
            res: fst that converts the verbalization of a number to that number
        """

        digits = defaultdict(list)
        ties = defaultdict(list)
        for digit_path in digit_paths:
            for k, v in load_labels(digit_path):
                digits[v].append(k)

        for k, v in load_labels(tie_path):
            ties[v].append(k)

        d = []
        for i in range(21, 100):
            s = str(i)
            if s[1] == "0":
                continue

            for di in digits[s[1]]:
                for ti in ties[s[0]]:
                    for and_word in and_words:
                        for before in ("", " "):
                            for after in ("", " "):
                                word = di + before + and_word + after + ti
                                d.append((word, s))

        res = pynini.string_map(d)
        return res

    def get_quantity(self, quantities: 'pynini.FstLike', written: str) -> 'pynini.FstLike':
        """
        getting all spoken forms of a quantity, e.g. "million", "millionen"

        Args:
            quantities: fst mapping the spoken forms of the quantities to their written forms
            written: written form of the quantity, e.g. 1.000.000
        Returns:
            res: acceptor for the spoken forms of that quantity
        """

        return pynini.project(quantities @ pynini.accep(written), "input").optimize()

    def get_magnitudes(self, quantities_path: str) -> List[str]:
        """
        getting the written forms of the quantities that span whole groups of three digits,
        ordered from the smallest to the largest

        Args:
            quantities_path: file to the quantities tsv, mapping the spoken forms of the quantities to
                their written forms, e.g. "million" -> 1.000.000
        Returns:
            res: written forms of the quantities, sorted by the number of digits they span
        """

        written_forms = {written for _, written in load_labels(quantities_path)}
        magnitudes = {written for written in written_forms if "." in written}
        return sorted(magnitudes, key=lambda written: len(written.replace(".", "")))
