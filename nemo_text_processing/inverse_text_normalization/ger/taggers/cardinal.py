# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
)


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinal numbers, e.g.
        minus eine billion fünfundsechzig milliarden vier millionen sechs -> cardinal { negative: '-' integer: "1.065.004.000.006" }
    The transducer implements a period separator every three digits by default.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # WFST mappings for numbers 0-99
        zero = pynini.string_map([("null", "0")])
        digits = pynini.string_file(get_abs_path("data/cardinal/digits.tsv"))
        # Isolates single digit cardinals to pass to other graphs
        self.digits = digits.optimize()
        to_denormalize = pynini.string_file(
            get_abs_path("data/cardinal/denormalized.tsv")
        )
        # Isolates the first dozen
        self.dozen = to_denormalize.optimize()
        teens = pynini.string_file(get_abs_path("data/cardinal/teens.tsv"))
        tens = pynini.string_file(get_abs_path("data/cardinal/tens.tsv"))
        ties = pynini.string_file(get_abs_path("data/cardinal/ties.tsv"))
        # German flips ones and tens in two-digit numbers. The WFST below handles these flips.
        flips = pynini.string_file(get_abs_path("data/cardinal/flips.tsv"))
        delete_space = pynutil.delete(NEMO_SPACE)
        delete_und = pynutil.delete("und")

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

        hundert = pynini.accep("hundert") | pynini.accep("ein hundert")
        hundreds = (pynini.cross(hundert, "100")) | (
            (
                (digits | pynutil.insert("1"))
                + delete_space.ques
                + pynutil.delete("hundert")
                + delete_space.ques
                + delete_und.ques
                + delete_space.ques
                + graph_10_99
            )
            | (
                (digits | pynutil.insert("1"))
                + delete_space.ques
                + pynini.cross("hundert", "0")
                + delete_space.ques
                + delete_und.ques
                + delete_space.ques
                + digits
            )
            | (
                (digits | pynutil.insert("1"))
                + delete_space.ques
                + pynini.cross("hundert", "00")
            )
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
        non_zero_digit_cluster = (
            (hundreds)
            | (pynutil.insert("0") + graph_10_99)
            | (pynutil.insert("00") + digits)
        )

        # WFST grammar for thousands
        thousands = (pynini.cross("tausend", "1.000")) | (
            (
                (pynini.cross("tausend", "1.") + delete_space.ques + delete_und.ques)
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross("tausend", ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + digit_cluster
        )

        non_zero_thousands = (pynini.cross("tausend", "1.000")) | (
            (
                (pynini.cross("tausend", "1.") + delete_space.ques + delete_und.ques)
                | (
                    non_zero_digit_cluster
                    + delete_space.ques
                    + pynini.cross("tausend", ".")
                    + delete_und.ques
                )
                # | pynutil.insert("000.")
            )
            + delete_space.ques
            + digit_cluster
        )

        # WFST grammar for millions
        million = pynini.accep("million") | pynini.accep("millionen")
        millions = (pynini.cross("million", "1.000.000")) | (
            (
                (pynini.cross("million", "1.") + delete_space.ques + delete_und.ques)
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross(million, ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + thousands
        )

        # WFST grammar for billions
        billion = (
            pynini.accep("milliarde")
            | pynini.accep("milliarden")
            # include the consonant-final stem for ordinal declensions e.g "milliardste"
            # "e" -> "" / _[ordinal morpheme]
            | pynini.accep("milliard")
        )
        billions = (pynini.cross("milliarde", "1.000.000.000")) | (
            (
                (
                    pynini.cross(
                        (pynini.accep("milliarde") | pynini.accep("milliard")), "1."
                    )
                    + delete_space.ques
                    + delete_und.ques
                )
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross(billion, ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + millions
        )

        # WFST grammar for trillions
        trillion = pynini.accep("billion") | pynini.accep("billionen")
        trillions = (pynini.cross("billion", "1.000.000.000.000")) | (
            (
                (pynini.cross("billion", "1.") + delete_space.ques + delete_und.ques)
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross(trillion, ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + billions
        )

        # WFST grammar for quadrillions
        quadrillion = (
            pynini.accep("billiarde")
            | pynini.accep("billiarden")
            # include the consonant-final stem for ordinal declensions e.g "billiardste"
            # "e" -> "" / _[ordinal morpheme]
            | pynini.accep("billiard")
        )
        quadrillions = (pynini.cross("billiarde", "1.000.000.000.000.000")) | (
            (
                (pynini.cross(quadrillion, "1.") + delete_space.ques + delete_und.ques)
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross(quadrillion, ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + trillions
        )

        # WFST grammar for quintillions
        quintillion = pynini.accep("trillion") | pynini.accep("trillionen")
        quintillions = (pynini.cross("trillion", "1.000.000.000.000.000.000")) | (
            (
                (pynini.cross("trillion", "1.") + delete_space.ques + delete_und.ques)
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross(quintillion, ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + quadrillions
        )

        # WFST grammar for sextillions
        sextillion = (
            pynini.accep("trilliarde")
            | pynini.accep("trilliarden")
            # include the consonant-final stem for ordinal declensions e.g "trilliardste"
            # "e" -> "" / _[ordinal morpheme]
            | pynini.accep("trilliard")
        )
        sextillions = (pynini.cross("billiarde", "1.000.000.000.000.000.000.000")) | (
            (
                (pynini.cross(sextillion, "1.") + delete_space.ques + delete_und.ques)
                | (
                    digit_cluster
                    + delete_space.ques
                    + pynini.cross(sextillion, ".")
                    + delete_und.ques
                )
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + quintillions
        )

        # Remove the leading zeros
        non_zero_digits = pynini.difference(NEMO_DIGIT, "0")
        chars_to_remove = pynini.accep("0") | pynini.accep(".")
        remove_chars = pynutil.delete(pynini.closure(chars_to_remove))
        remove_leading_zeros = pynini.cdrewrite(
            remove_chars, "[BOS]", non_zero_digits, NEMO_SIGMA
        )

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
        single_digit_years = pynini.string_file(
            get_abs_path("data/cardinal/single_digit_years.tsv")
        )
        ten = "zehn"
        remove_ten = pynini.project(graph_10_99, "input") - ten
        graph_11_99 = remove_ten @ graph_10_99

        years_exceptions = (
            graph_11_99
            + pynutil.delete(NEMO_SPACE).ques
            + pynutil.delete("hundert").ques
            + pynutil.delete(NEMO_SPACE).ques
            + (single_digit_years | graph_10_99 | pynutil.insert("00"))
        )
        years = first_millenium | second_tenth_millenium | years_exceptions
        remove_period_separators = pynini.cdrewrite(
            pynutil.delete("."), "", "", NEMO_SIGMA
        )
        years = years @ remove_leading_zeros @ remove_period_separators
        self.graph_years = years.optimize()

        # The block below leaves numerals 1 - 12 canonically normalized
        accept_denormalized_first_dozen = pynini.project(
            to_denormalize, "input"
        )  # acceptor for null - zwölf
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
            pynutil.insert("negative: ")
            + pynini.cross("minus ", '"-"')
            + pynutil.insert(" "),
            0,
            1,
        )

        all_cardinals_graph = (
            self.optional_negative
            + pynutil.insert('integer: "')
            + self.graph_all_cardinals
            + pynutil.insert('"')
        )
        self.all_cardinals_graph = all_cardinals_graph.optimize()

        # The final graph for this semiotic class leaves the first dozen normalized
        final_graph = (
            self.optional_negative
            + pynutil.insert('integer: "')
            + self.graph
            + pynutil.insert('"')
        )

        # The block below handles noun + number combinations, where the noun forces full denormalization
        # The nouns are implemented as a .tsv list
        nouns_forcing_denormalization = pynini.string_file(
            get_abs_path("data/measure/nouns_forcing_denormalization.tsv")
        )
        graph_forced_denormalization = (
            pynutil.insert("morphosyntactic_features: ")
            + pynutil.insert('"')
            + nouns_forcing_denormalization
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + all_cardinals_graph
        )

        # Canonical representation with the first dozen normalized
        self.canonical_cardinals_graph = final_graph.optimize()

        # Updated final graph to account for DPs with a denormalization-inducing noun in the D' position
        updated_final_graph = (final_graph | graph_forced_denormalization).optimize()

        final_graph = self.add_tokens(updated_final_graph)
        self.fst = final_graph.optimize()
