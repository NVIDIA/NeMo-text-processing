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


import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_DIGIT, NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


def _digit_tie_flips():
    """Map concatenated ones+tens (12 for einundzwanzig) to the written number (21)."""
    return pynini.string_map([(f"{ones}{tens}", f"{tens}{ones}") for tens in range(2, 10) for ones in range(1, 10)])


def _forms(lexicon, lemma):
    """Spoken forms for one lemma (e.g. million / millionen)."""
    return pynini.project(lexicon @ pynini.accep(lemma), "input").optimize()


def _token(table, name):
    """Output string from format.tsv."""
    return pynini.project(pynini.accep(name) @ table, "output").optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below ten are not converted.
    Allows both compound numeral strings or separated by whitespace.
    "und" (en: "and") can be inserted between "hundert" and following number or "tausend" and following single or double digit number.

        e.g. minus drei und zwanzig -> cardinal { negative: "true" integer: "23" }
        e.g. minus dreiundzwanzig -> cardinal { negative: "true" integer: "23" }
        e.g. dreizehn -> cardinal { integer: "13" }
        e.g. ein hundert -> cardinal { integer: "100" }
        e.g. einhundert -> cardinal { integer: "100" }
        e.g. ein tausend -> cardinal { integer: "1.000" }
        e.g. eintausend -> cardinal { integer: "1.000" }
        e.g. ein tausend zwanzig -> cardinal { integer: "1.020" }
        e.g. minus eine billion fünfundsechzig milliarden vier millionen sechs -> cardinal { negative: "true" integer: "1.065.004.000.006" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # WFST mappings for numbers 0-99
        zero = pynini.string_file(get_abs_path("data/cardinal/zero.tsv"))
        digits = pynini.string_file(get_abs_path("data/cardinal/digits.tsv"))
        # Isolates single digit cardinals to pass to other graphs
        self.digits = digits.optimize()
        irregular_teens = pynini.string_file(get_abs_path("data/cardinal/irregular_teens.tsv"))
        # 0-12 stay as words: zero + digits (1-9) + irregular teens (10-12)
        to_denormalize = zero | digits | irregular_teens
        # Isolates the first dozen
        self.dozen = to_denormalize.optimize()
        teens = pynini.string_file(get_abs_path("data/cardinal/teens.tsv"))
        tens = pynini.string_file(get_abs_path("data/cardinal/tens.tsv"))
        # Standalone decades: tens digit (2) + 0 -> 20
        ties = tens + pynutil.insert("0")
        # German flips ones and tens in two-digit numbers (ein + zwanzig -> 21).
        flips = _digit_tie_flips()
        lexicon = pynini.string_file(get_abs_path("data/cardinal/lexicon.tsv"))
        und = _forms(lexicon, "und")
        minus = _forms(lexicon, "minus")
        hundert = _forms(lexicon, "hundert")
        tausend = _forms(lexicon, "tausend")
        million = _forms(lexicon, "million")
        milliarde = _forms(lexicon, "milliarde")
        billion_de = _forms(lexicon, "billion")
        billiarde = _forms(lexicon, "billiarde")
        trillion_de = _forms(lexicon, "trillion")
        trilliarde = _forms(lexicon, "trilliarde")
        mag = pynini.string_file(get_abs_path("data/cardinal/magnitude.tsv"))
        fmt = pynini.string_file(get_abs_path("data/cardinal/format.tsv"))
        lead = _token(fmt, "lead")
        dot = _token(fmt, "dot")
        n000dot = _token(fmt, "n000dot")
        delete_space = pynutil.delete(NEMO_SPACE)
        delete_und = pynutil.delete(und)

        # Accepts normalized digits+ties (ein+und+zwanzig)
        digit_ties = digits + delete_space.ques + delete_und + delete_space.ques + tens
        # Flips ties and digits for denormalization
        ties_digit = digit_ties @ flips

        # WFST grammar for hundreds
        graph_10_99 = teens | ties | ties_digit
        self.graph_double_digits = graph_10_99
        # Isolates single and double-digit cardinals to pass to other graphs
        graph_single_and_double_digits = digits | graph_10_99
        self.graph_single_and_double_digits = graph_single_and_double_digits.optimize()

        h = pynini.string_file(get_abs_path("data/cardinal/hundred.tsv"))
        hundred = h @ pynini.accep("100")
        hundred_0 = h @ pynini.accep("0")
        hundred_00 = h @ pynini.accep("00")
        hundreds = (hundred) | (
            (
                (digits | pynutil.insert("1"))
                + delete_space.ques
                + pynutil.delete(hundert)
                + delete_space.ques
                + delete_und.ques
                + delete_space.ques
                + graph_10_99
            )
            | (
                (digits | pynutil.insert("1"))
                + delete_space.ques
                + hundred_0
                + delete_space.ques
                + delete_und.ques
                + delete_space.ques
                + digits
            )
            | ((digits | pynutil.insert("1")) + delete_space.ques + hundred_00)
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

        # WFST grammar for thousands
        thousands = (tausend @ mag) | (
            (
                (pynini.cross(tausend, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(tausend, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + digit_cluster
        )

        non_zero_thousands = (tausend @ mag) | (
            (
                (pynini.cross(tausend, lead) + delete_space.ques + delete_und.ques)
                | (non_zero_digit_cluster + delete_space.ques + pynini.cross(tausend, dot) + delete_und.ques)
                # | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + digit_cluster
        )

        # WFST grammar for millions
        millions = (million @ mag) | (
            (
                (pynini.cross(million, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(million, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + thousands
        )

        # WFST grammar for billions
        billion = milliarde
        billions = (milliarde @ mag) | (
            (
                (pynini.cross(milliarde, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(billion, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + millions
        )

        # WFST grammar for trillions
        trillion = billion_de
        trillions = (billion_de @ mag) | (
            (
                (pynini.cross(billion_de, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(trillion, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + billions
        )

        # WFST grammar for quadrillions
        quadrillion = billiarde
        quadrillions = (billiarde @ mag) | (
            (
                (pynini.cross(quadrillion, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(quadrillion, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + trillions
        )

        # WFST grammar for quintillions
        quintillion = trillion_de
        quintillions = (trillion_de @ mag) | (
            (
                (pynini.cross(trillion_de, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(quintillion, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + quadrillions
        )

        # WFST grammar for sextillions
        sextillion = trilliarde
        sextillions = (trilliarde @ mag) | (
            (
                (pynini.cross(sextillion, lead) + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(sextillion, dot) + delete_und.ques)
                | pynutil.insert(n000dot)
            )
            + delete_space.ques
            + quintillions
        )

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
            + pynutil.delete(hundert).ques
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
            pynutil.insert("negative: ") + pynini.cross(minus + pynini.accep(" "), '"true"') + pynutil.insert(" "),
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
