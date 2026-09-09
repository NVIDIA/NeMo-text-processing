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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_DIGIT, NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, load_labels

HUNDRED = "hundert"
THOUSAND = "tausend"
MILLION = "million"
BILLION = "milliarde"
TRILLION = "billion"
QUADRILLION = "billiarde"
QUINTILLION = "trillion"
SEXTILLION = "trilliarde"


def get_ties_digit(digit_path: str, tie_path: str, and_word: str) -> 'pynini.FstLike':
    """
    getting all denormalizations for numbers between 21 - 100

    Args:
        digit_path: file to digit tsv
        tie_path: file to tie tsv, e.g. 20, 30, etc.
        and_word: connector between the digit and the tie, e.g. "und"
    Returns:
        res: fst that converts the verbalization of a number to that number
    """

    digits = defaultdict(list)
    ties = defaultdict(list)
    for k, v in load_labels(digit_path):
        digits[v].append(k)
    digits["1"] = ["ein"]

    for k, v in load_labels(tie_path):
        ties[v].append(k)

    d = []
    for i in range(21, 100):
        s = str(i)
        if s[1] == "0":
            continue

        for di in digits[s[1]]:
            for ti in ties[s[0]]:
                # both the compound spelling and the spaced one are attested, e.g. "einundzwanzig", "ein und zwanzig"
                for before in ("", " "):
                    for after in ("", " "):
                        word = di + before + and_word + after + ti
                        d.append((word, s))

    res = pynini.string_map(d)
    return res


def get_quantity(word: str, plural: str) -> 'pynini.FstLike':
    """
    getting the singular and the plural spoken form of a quantity

    Args:
        word: singular form of the quantity, e.g. "million"
        plural: plural suffix of the quantity, e.g. "en"
    Returns:
        res: acceptor for both spoken forms, e.g. "million", "millionen"
    """

    return pynini.accep(word) + pynini.accep(plural).ques


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below ten are not converted.
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
        e.g. minus eine billion fünfundsechzig milliarden vier millionen sechs -> cardinal { negative: "-" integer: "1.065.004.000.006" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # WFST mappings for numbers 0-99
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit_no_one = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_one = pynini.string_file(get_abs_path("data/numbers/ones.tsv"))
        digits = graph_digit_no_one | graph_one
        # Isolates single digit cardinals to pass to other graphs
        self.digits = digits.optimize()
        teens = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        # Numerals up to twelve are spelled out, so they are kept apart from the rest of the teens
        irregular_teens = teens @ pynini.union("10", "11", "12")
        # 0-12 stay as words: zero + digits (1-9) + irregular teens (10-12)
        to_denormalize = zero | digits | irregular_teens
        # Isolates the first dozen
        self.dozen = to_denormalize.optimize()
        tens = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        # Standalone decades: tens digit (2) + 0 -> 20
        ties = tens + pynutil.insert("0")
        and_word = load_labels(get_abs_path("data/numbers/und.tsv"))[0][0]
        minus = pynini.string_file(get_abs_path("data/numbers/minus.tsv"))
        # German flips ones and tens in two-digit numbers, e.g. "ein und zwanzig" -> 21
        ties_digit = get_ties_digit(
            get_abs_path("data/numbers/digit.tsv"), get_abs_path("data/numbers/ties.tsv"), and_word
        )
        delete_space = pynutil.delete(NEMO_SPACE)
        delete_und = pynutil.delete(and_word)

        # WFST grammar for hundreds
        graph_10_99 = teens | ties | ties_digit
        self.graph_double_digits = graph_10_99
        # Isolates single and double-digit cardinals to pass to other graphs
        graph_single_and_double_digits = digits | graph_10_99
        self.graph_single_and_double_digits = graph_single_and_double_digits.optimize()

        # "hundert" is preceded by an optional multiplier and followed by the two digits it leaves empty
        hundert = pynutil.delete(HUNDRED)
        multiplier = (digits | pynutil.insert("1")) + delete_space.ques
        hundreds = (
            (multiplier + hundert + delete_space.ques + delete_und.ques + delete_space.ques + graph_10_99)
            | (
                multiplier
                + hundert
                + pynutil.insert("0")
                + delete_space.ques
                + delete_und.ques
                + delete_space.ques
                + digits
            )
            | (multiplier + hundert + pynutil.insert("00"))
        )

        # Digits are grouped in clusters of three: {hundreds}{tens}{ones}.
        # Clusters of three are separated by periods, applied right to left.
        digit_cluster = (
            (hundreds)
            | (pynutil.insert("0") + graph_10_99)
            | (pynutil.insert("00") + digits)
            | (pynutil.insert("000"))
        )
        # The subgraph below introduces three-digit clusters containing at least one non-zero digit.
        # It is mainly utilized by the "years" subgraph in the DATE class.
        non_zero_digit_cluster = (hundreds) | (pynutil.insert("0") + graph_10_99) | (pynutil.insert("00") + digits)

        def magnitude(quantity, groups, lower_magnitudes, leading_cluster=digit_cluster, empty_multiplier=True):
            """
            WFST grammar for one order of magnitude, e.g. "million"

            Args:
                quantity: acceptor for the spoken forms of the quantity, e.g. "million", "millionen"
                groups: number of three-digit clusters below this magnitude, e.g. 2 for "million"
                lower_magnitudes: WFST grammar for the next magnitude down, e.g. "thousands"
                leading_cluster: WFST grammar for the cluster multiplying the quantity
                empty_multiplier: whether the magnitude may be skipped, e.g. "eine million drei"
            """
            multiplied = pynutil.delete(quantity) + pynutil.insert("1.") + delete_space.ques + delete_und.ques | (
                leading_cluster + delete_space.ques + pynutil.delete(quantity) + pynutil.insert(".") + delete_und.ques
            )
            if empty_multiplier:
                multiplied |= pynutil.insert("000.")
            # The quantity on its own, e.g. "million" -> 1.000.000
            standalone = pynutil.delete(quantity) + pynutil.insert("1" + ".000" * groups)
            return standalone | (multiplied + delete_space.ques + lower_magnitudes)

        thousands = magnitude(THOUSAND, 1, digit_cluster)
        non_zero_thousands = magnitude(
            THOUSAND, 1, digit_cluster, leading_cluster=non_zero_digit_cluster, empty_multiplier=False
        )
        millions = magnitude(get_quantity(MILLION, "en"), 2, thousands)
        billions = magnitude(get_quantity(BILLION, "n"), 3, millions)
        trillions = magnitude(get_quantity(TRILLION, "en"), 4, billions)
        quadrillions = magnitude(get_quantity(QUADRILLION, "n"), 5, trillions)
        quintillions = magnitude(get_quantity(QUINTILLION, "en"), 6, quadrillions)
        sextillions = magnitude(get_quantity(SEXTILLION, "n"), 7, quintillions)

        # Remove the leading zeros
        non_zero_digits = pynini.difference(NEMO_DIGIT, "0")
        chars_to_remove = pynini.accep("0") | pynini.accep(".")
        remove_chars = pynutil.delete(pynini.closure(chars_to_remove))
        remove_leading_zeros = pynini.cdrewrite(remove_chars, "[BOS]", non_zero_digits, NEMO_SIGMA)

        # All together now
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

        # Generates a graph accepting all digits to be passed to other semiotic classes
        graph_everything = graph_cardinals @ remove_leading_zeros
        self.graph_all_cardinals = graph_everything.optimize()

        # Generates a graph denormalizing years from 0 to 9999
        # The graph will be passed into other semiotic classes
        # Years 0 - 999 denormalize as regular cardinals
        first_millenium = non_zero_digit_cluster  # | zero
        second_tenth_millenium = non_zero_thousands
        # The graph below covers exceptions
        # e.g. years 1100 - 1999
        # and all colloquial expresions (e.g. zwanzigvierundzwanzig -> 2024)
        ten = pynini.project(irregular_teens @ pynini.accep("10"), "input")
        remove_ten = pynini.project(graph_10_99, "input") - ten
        graph_11_99 = remove_ten @ graph_10_99

        years_exceptions = (
            graph_11_99
            + pynutil.delete(NEMO_SPACE).ques
            + pynutil.delete(HUNDRED).ques
            + pynutil.delete(NEMO_SPACE).ques
            + (graph_10_99 | pynutil.insert("00"))
        )
        years = first_millenium | second_tenth_millenium | years_exceptions
        remove_period_separators = pynini.cdrewrite(pynutil.delete("."), "", "", NEMO_SIGMA)
        years = years @ remove_leading_zeros @ remove_period_separators
        self.graph_years = years.optimize()

        # The block below leaves numerals 1 - 12 canonically normalized
        accept_denormalized_first_dozen = pynini.project(to_denormalize, "input")  # acceptor for null - zwölf
        accept_denormalized_everything = pynini.project(
            self.graph_all_cardinals, "input"
        )  # acceptor for all verbalized cardinals
        accept_without_first_dozen = (
            accept_denormalized_everything - accept_denormalized_first_dozen
        )  # acceptor for all verbalized cardinals greater than 12
        transduce_without_first_dozen = (
            accept_without_first_dozen @ self.graph_all_cardinals
        )  # transducer for all verbalized cardinals greater than 12
        graph = accept_denormalized_first_dozen | transduce_without_first_dozen
        self.graph = graph.optimize()

        self.optional_negative = pynini.closure(
            pynutil.insert('negative: "') + minus + pynutil.delete(" ") + pynutil.insert('" '),
            0,
            1,
        )

        all_cardinals_graph = (
            self.optional_negative + pynutil.insert('integer: "') + self.graph_all_cardinals + pynutil.insert('"')
        )
        self.all_cardinals_graph = all_cardinals_graph.optimize()

        # The final graph for this semiotic class leaves the first dozen normalized
        final_graph = self.optional_negative + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')

        # Canonical representation with the first dozen normalized
        self.canonical_cardinals_graph = final_graph.optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

        self.graph_no_exception = self.graph_all_cardinals
        self.optional_minus_graph = self.optional_negative
        self.graph_hundred_component_at_least_one_none_zero_digit = self.graph_all_cardinals
        self.digit = self.digits
        self.graph_ties = self.graph_double_digits
