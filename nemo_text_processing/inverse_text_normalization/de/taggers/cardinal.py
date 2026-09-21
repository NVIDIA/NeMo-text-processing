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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, load_labels

# ordered by value, each scale is three decimal digits larger than the previous one
MAGNITUDE_SCALES = ("hundert", "tausend", "million", "milliarde", "billion", "billiarde", "trillion", "trilliarde")
# after a bare integer these stay with the cardinal grammar: zwei tausend -> 2.000
CARDINAL_SCALES = ("hundert", "tausend")


def get_tens_digit(digit_path: str, tens_path: str, conjunction_path: str) -> 'pynini.FstLike':
    """
    getting all denormalizations for numbers between 21 - 99. German says the ones digit
    before the tens digit (ein-und-zwanzig = 21), so the words cannot be read left to right

    Args:
        digit_path: file to digits tsv
        tens_path: file to tens tsv, e.g. zwanzig -> 2
        conjunction_path: file to the conjunction tsv, e.g. und
    Returns:
        res: fst that converts the verbalization of a number to its digits
    """

    conjunction = load_labels(conjunction_path)[0][0]
    digits = defaultdict(list)
    ties = defaultdict(list)
    for k, v in load_labels(digit_path):
        digits[v].append(k)

    for k, v in load_labels(tens_path):
        ties[v].append(k)

    d = []
    for i in range(21, 100):
        s = str(i)
        if s[1] == "0":
            continue

        for di in digits[s[1]]:
            for ti in ties[s[0]]:
                word = di + conjunction + ti
                d.append((word, s))

    res = pynini.string_map(d)
    return res


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below thirteen are not converted
    unless they carry a minus sign.
    Allows both compound numeral strings or separated by whitespace.
    "und" (en: "and") after "hundert", "tausend" or a larger magnitude word is never part of the
    number, it joins two numbers, whether written glued or spaced: both "einhundertundzwei" and
    "ein hundert und zwei" -> 100 und 2. German writes 102 as "ein hundert zwei".

        e.g. minus drei und zwanzig -> cardinal { negative: "-" integer: "23" }
        e.g. minus dreiundzwanzig -> cardinal { negative: "-" integer: "23" }
        e.g. dreizehn -> cardinal { integer: "13" }
        e.g. ein hundert -> cardinal { integer: "100" }
        e.g. einhundert -> cardinal { integer: "100" }
        e.g. ein tausend -> cardinal { integer: "1.000" }
        e.g. eintausend -> cardinal { integer: "1.000" }
        e.g. ein tausend zwanzig -> cardinal { integer: "1.020" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        digits = pynini.string_file(get_abs_path("data/numbers/digits.tsv"))
        irregular_teens = pynini.string_file(get_abs_path("data/numbers/irregular_teens.tsv"))
        to_denormalize = zero | digits | irregular_teens

        regular_teens = pynini.string_file(get_abs_path("data/numbers/regular_teens.tsv"))
        teens = irregular_teens | regular_teens
        tens = pynini.string_file(get_abs_path("data/numbers/tens.tsv"))
        ties = tens + pynutil.insert("0")
        conjunction = load_labels(get_abs_path("data/numbers/conjunction.tsv"))[0][0]

        # the map is keyed on the compound spelling, so whitespace is stripped before lookup
        delete_all_spaces = pynini.cdrewrite(pynutil.delete(NEMO_WHITE_SPACE), "", "", NEMO_SIGMA)
        ties_digit = delete_all_spaces @ get_tens_digit(
            get_abs_path("data/numbers/digits.tsv"),
            get_abs_path("data/numbers/tens.tsv"),
            get_abs_path("data/numbers/conjunction.tsv"),
        )

        graph_10_99 = teens | ties | ties_digit

        self.magnitude = pynini.string_file(get_abs_path("data/numbers/quantity.tsv")).optimize()
        self.magnitude_words = pynini.project(self.magnitude, "input").optimize()
        # the scale name is a prefix of all its own forms and of no other scale
        self.scale_forms = {
            scale: pynini.intersect(self.magnitude_words, pynini.accep(scale) + NEMO_SIGMA).optimize()
            for scale in MAGNITUDE_SCALES
        }

        hundert = self.scale_forms["hundert"]
        hundreds = (
            ((digits | pynutil.insert("1")) + delete_space + pynutil.delete(hundert) + delete_space + graph_10_99)
            | ((digits | pynutil.insert("1")) + delete_space + pynini.cross(hundert, "0") + delete_space + digits)
            | ((digits | pynutil.insert("1")) + delete_space + pynini.cross(hundert, "00"))
        )

        # Digits are grouped in clusters of three: {hundreds}{tens}{ones}.
        non_zero_digit_cluster = (hundreds) | (pynutil.insert("0") + graph_10_99) | (pynutil.insert("00") + digits)
        digit_cluster = non_zero_digit_cluster | pynutil.insert("000")
        # a magnitude word with no multiplier in front of it means "one" of that magnitude
        leading_cluster = non_zero_digit_cluster | pynutil.insert("001")

        thousands = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["tausend"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + digit_cluster
        )

        millions = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["million"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + thousands
        )

        billions = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["milliarde"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + millions
        )

        trillions = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["billion"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + billions
        )

        quadrillions = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["billiarde"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + trillions
        )

        quintillions = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["trillion"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + quadrillions
        )

        sextillions = (
            (
                (leading_cluster + delete_space + pynini.cross(self.scale_forms["trilliarde"], "."))
                | pynutil.insert("000.")
            )
            + delete_space
            + quintillions
        )

        non_zero_digits = pynini.difference(NEMO_DIGIT, "0")
        chars_to_remove = pynini.accep("0") | pynini.accep(".")
        remove_chars = pynutil.delete(pynini.closure(chars_to_remove))
        remove_leading_zeros = pynini.cdrewrite(remove_chars, "[BOS]", non_zero_digits, NEMO_SIGMA)

        grammars = [
            sextillions,
            quintillions,
            quadrillions,
            trillions,
            billions,
            millions,
            thousands,
            digit_cluster,
            zero,
        ]

        graph_cardinals = ""
        for grammar in grammars:
            graph_cardinals |= grammar

        # the graph the other German semiotic classes consume, without the first-dozen exception
        self.graph_no_exception = (graph_cardinals @ remove_leading_zeros).optimize()

        # 1-999 without leading zeros, consumed by the decimal tagger's get_quantity
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            non_zero_digit_cluster @ remove_leading_zeros
        ).optimize()

        # the block below leaves numerals 1 - 12 spelled out
        accept_denormalized_first_dozen = pynini.project(to_denormalize, "input")
        accept_denormalized_everything = pynini.project(self.graph_no_exception, "input")
        accept_without_first_dozen = accept_denormalized_everything - accept_denormalized_first_dozen
        transduce_without_first_dozen = accept_without_first_dozen @ self.graph_no_exception
        graph = accept_denormalized_first_dozen | transduce_without_first_dozen
        self.graph = graph.optimize()

        ends_in_magnitude = pynini.compose(NEMO_SIGMA + self.magnitude_words, self.graph_no_exception)
        graph_magnitude_und = (
            ends_in_magnitude
            + delete_space
            + pynutil.insert(" ")
            + pynini.accep(conjunction)
            + pynutil.insert(" ")
            + delete_space
            + self.graph_hundred_component_at_least_one_none_zero_digit
        )

        negative = pynutil.insert("negative: ") + pynini.cross("minus ", '"-"') + pynutil.insert(" ")
        self.optional_minus_graph = pynini.closure(negative, 0, 1)
        integer = pynutil.insert('integer: "') + (self.graph | graph_magnitude_und) + pynutil.insert('"')
        # a sign in front of zero carries no meaning, so "minus null" is not a cardinal
        accept_zero = pynini.project(zero, "input")
        graph_no_exception_non_zero = (
            pynini.difference(accept_denormalized_everything, accept_zero) @ self.graph_no_exception
        )

        negative_integer = (
            negative
            + pynutil.insert('integer: "')
            + (graph_no_exception_non_zero | graph_magnitude_und)
            + pynutil.insert('"')
        )

        final_graph = integer | negative_integer

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
