# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_SPACE,
    NEMO_TE_DIGIT,
    GraphFst,
    delete_space,
    integer_to_telugu,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path, load_labels


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu cardinals.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()

        graph_digit_people = pynini.string_file(get_abs_path("data/numbers/digit_people.tsv"))

        graph_digit |= graph_digit_people
        graph_digit |= pynini.cross("ఒక", "౧")

        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()

        graph_special = pynini.string_file(get_abs_path("data/numbers/special_numbers.tsv"))

        graph_teens_and_ties |= graph_special

        self.graph_zero = graph_zero
        self.graph_digit = graph_digit
        graph_leading_zero_sequence = graph_zero + pynini.closure(
            delete_space + (graph_zero | graph_digit),
            1,
        )
        graph_ties_prefix = pynini.string_file(get_abs_path("data/numbers/ties_prefix.tsv"))

        graph_okkati = pynini.cross("ఒకటి", "౧")
        graph_ties_plus_okkati = graph_ties_prefix + delete_space + graph_okkati

        self.graph_single_digit_with_zero = pynutil.insert("౦") + graph_digit

        self.graph_two_digit = (
            graph_teens_and_ties
            | graph_ties_plus_okkati
            | (graph_ties_prefix + delete_space + graph_digit)
            | self.graph_single_digit_with_zero
        )
        hundred_words = [row[0] for row in load_labels(get_abs_path("data/numbers/hundred_words.tsv"))]
        thousand_words = [row[0] for row in load_labels(get_abs_path("data/numbers/thousand_words.tsv"))]
        lakh_words = [row[0] for row in load_labels(get_abs_path("data/numbers/lakh_words.tsv"))]
        crore_words = [row[0] for row in load_labels(get_abs_path("data/numbers/crore_words.tsv"))]
        graph_hundred_word = pynini.union(*hundred_words)
        delete_hundred = pynutil.delete(graph_hundred_word)

        graph_hundreds_component = pynini.union(
            graph_digit + delete_space + delete_hundred,
            pynutil.insert("౦"),
        )
        graph_hundreds_component += delete_space
        graph_hundreds_component += self.graph_two_digit | pynutil.insert("౦౦")

        graph_hundred_standalone = pynini.union(*[pynini.cross(word, "౧౦౦") for word in hundred_words])

        two_digit_hundred_values = pynini.string_map(
            [(integer_to_telugu(i), integer_to_telugu(i * 100)) for i in range(10, 100)]
        ).optimize()

        graph_hundred_as_thousand = (self.graph_two_digit @ two_digit_hundred_values) + delete_space + delete_hundred

        graph_nuta_component = (
            pynutil.delete(pynini.union("నూట", "వంద")) + pynutil.insert("౧") + delete_space + self.graph_two_digit
        )
        self.graph_hundreds = (
            graph_hundred_as_thousand | graph_hundreds_component | graph_hundred_standalone | graph_nuta_component
        )

        self.graph_hundred_component_at_least_one_none_zero_digit = graph_hundreds_component @ (
            pynini.closure(NEMO_TE_DIGIT) + (NEMO_TE_DIGIT - "౦") + pynini.closure(NEMO_TE_DIGIT)
        )

        graph_below_thousand = self.graph_hundreds | self.graph_two_digit | graph_digit

        graph_remainder_three_digit = (
            self.graph_hundreds | (pynutil.insert("౦") + self.graph_two_digit) | pynutil.insert("౦౦౦")
        )

        graph_group_two_digit_leading = self.graph_two_digit | graph_digit
        graph_group_two_digit_padded = (
            self.graph_two_digit | (pynutil.insert("౦") + graph_digit) | pynutil.insert("౦౦")
        )

        case_suffixes = [row[0] for row in load_labels(get_abs_path("data/numbers/case_suffix.tsv"))]

        graph_case_suffix = pynini.union(*case_suffixes)

        delete_thousand = pynini.union(
            *[pynutil.delete(word) for word in thousand_words],
            *[pynutil.delete(word + graph_case_suffix) for word in thousand_words],
        )
        delete_lakh = pynini.union(
            *[pynutil.delete(word) for word in lakh_words],
            *[pynutil.delete(word + graph_case_suffix) for word in lakh_words],
        )
        delete_crore = pynini.union(
            *[pynutil.delete(word) for word in crore_words],
            *[pynutil.delete(word + graph_case_suffix) for word in crore_words],
        )

        graph_thousands = (
            graph_group_two_digit_leading + delete_space + delete_thousand + delete_space + graph_remainder_three_digit
        )

        graph_thousands_padded = (
            graph_group_two_digit_padded + delete_space + delete_thousand + delete_space + graph_remainder_three_digit
        )

        graph_bare_thousand = (
            pynutil.delete(thousand_words[0]) + pynutil.insert("౧") + delete_space + graph_remainder_three_digit
        )

        graph_bare_thousand_terminal = pynutil.delete(thousand_words[0]) + pynutil.insert("౧౦౦౦")

        graph_lakhs = (
            graph_group_two_digit_leading
            + delete_space
            + delete_lakh
            + delete_space
            + graph_group_two_digit_padded
            + delete_space
            + delete_thousand
            + delete_space
            + graph_remainder_three_digit
        )

        graph_bare_lakh = (
            graph_group_two_digit_leading
            + delete_space
            + delete_lakh
            + delete_space
            + pynutil.insert("౦౦")
            + delete_space
            + graph_remainder_three_digit
        )

        graph_lakh_terminal = (
            pynini.union(
                pynini.cross("లక్ష", "౧"),
                pynini.cross("లక్షలు", "౧"),
                pynini.cross("లక్షల", "౧"),
            )
            | (graph_group_two_digit_leading + delete_space + delete_lakh)
        ) + pynutil.insert("౦౦౦౦౦")

        graph_crore_prefix_simple = (
            graph_group_two_digit_leading
            | self.graph_hundreds
            | graph_thousands
            | graph_bare_thousand
            | graph_lakhs
            | graph_bare_lakh
            | graph_lakh_terminal
        )

        graph_crore_terminal = pynutil.add_weight(
            (
                pynini.cross("కోటి", "౧")
                | pynini.cross("కోట్లు", "౧")
                | pynini.cross("కోట్ల", "౧")
                | (graph_crore_prefix_simple + delete_space + delete_crore)
            )
            + pynutil.insert("౦౦౦౦౦౦౦"),
            -0.1,
        )

        graph_bare_crore = (
            graph_crore_prefix_simple
            + delete_space
            + delete_crore
            + delete_space
            + pynutil.insert("౦౦")
            + delete_space
            + pynutil.insert("౦౦")
            + delete_space
            + graph_remainder_three_digit
        )

        graph_crores = (
            graph_crore_prefix_simple
            + delete_space
            + delete_crore
            + delete_space
            + graph_group_two_digit_padded
            + delete_space
            + delete_lakh
            + delete_space
            + graph_group_two_digit_padded
            + delete_space
            + delete_thousand
            + delete_space
            + graph_remainder_three_digit
        )

        graph_lower_seven = (
            graph_lakhs
            | graph_bare_lakh
            | graph_lakh_terminal
            | (pynutil.insert("౦౦") + graph_thousands_padded)
            | (pynutil.insert("౦౦౦") + graph_bare_thousand)
            | (pynutil.insert("౦౦౦") + graph_bare_thousand_terminal)
            | (pynutil.insert("౦౦౦౦") + graph_remainder_three_digit)
        )
        graph_one_lakh = (
            (pynutil.delete("లక్ష") | pynutil.delete("లక్షా"))
            + delete_space
            + (
                (pynutil.insert("౧౦౦౦౦") + graph_digit)
                | (pynutil.insert("౧౦౦౦") + self.graph_two_digit)
                | (pynutil.insert("౧౦౦") + self.graph_hundreds)
                | (pynutil.insert("౧౦") + graph_bare_thousand_terminal)
                | (pynutil.insert("౧") + graph_thousands)
                | (pynutil.insert("౧") + graph_thousands_padded)
            )
        )

        graph_one_crore = pynutil.add_weight(
            (
                pynutil.delete(crore_words[0])
                + delete_space
                + (
                    (pynutil.insert("౧౦౦౦౦౦౦") + graph_digit)
                    | (pynutil.insert("౧౦౦౦౦౦") + self.graph_two_digit)
                    | (pynutil.insert("౧౦౦౦౦") + self.graph_hundreds)
                    | (pynutil.insert("౧౦౦౦") + graph_bare_thousand_terminal)
                    | (pynutil.insert("౧౦౦") + graph_thousands)
                    | (pynutil.insert("౧౦౦") + graph_thousands_padded)
                    | (pynutil.insert("౧") + graph_lakhs)
                    | (pynutil.insert("౧") + graph_bare_lakh)
                    | (pynutil.insert("౧") + graph_lakh_terminal)
                )
            ),
            -10.0,
        )

        graph_large_crore_prefix = (
            graph_group_two_digit_leading
            | self.graph_hundreds
            | graph_thousands
            | graph_bare_thousand
            | graph_bare_thousand_terminal
            | graph_lakhs
            | graph_bare_lakh
            | graph_lakh_terminal
            | graph_crores
            | graph_bare_crore
            | graph_crore_terminal
        )

        graph_large_crores = (
            graph_large_crore_prefix
            + delete_space
            + delete_crore
            + (pynutil.insert("౦౦౦౦౦౦౦") | (delete_space + graph_lower_seven))
        )

        graph_no_prefix = pynutil.add_weight(
            pynini.union(
                *[pynini.cross(word, "౧౦౦") for word in hundred_words],
                *[pynini.cross(word, "౧౦౦౦") for word in thousand_words],
                *[pynini.cross(word, "౧౦౦౦౦౦") for word in lakh_words],
                *[pynini.cross(word, "౧౦౦౦౦౦౦౦") for word in crore_words],
            ),
            2,
        )

        graph_base = pynini.union(
            graph_large_crores,
            graph_crores,
            graph_bare_crore,
            graph_one_crore,
            graph_lakhs,
            graph_bare_lakh,
            graph_one_lakh,
            graph_lakh_terminal,
            graph_thousands,
            graph_bare_thousand,
            graph_bare_thousand_terminal,
            graph_zero,
            graph_no_prefix,
            graph_below_thousand,
        )

        graph_normal = graph_base | (graph_base + graph_case_suffix)

        graph_normal = graph_normal @ pynini.union(
            pynutil.delete(pynini.closure("౦"))
            + pynini.difference(NEMO_TE_DIGIT, "౦")
            + pynini.closure(NEMO_TE_DIGIT),
            "౦",
        )

        graph = graph_normal | graph_leading_zero_sequence

        labels_exception = [pynini.string_file(get_abs_path("data/numbers/labels_exception.tsv"))]
        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = graph
        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE,
            0,
            1,
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
