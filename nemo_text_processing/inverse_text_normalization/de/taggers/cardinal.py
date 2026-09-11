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


def swap_tens_and_ones(digits: 'pynini.FstLike', tens: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    German says the ones digit before the tens digit (ein-und-zwanzig = 21), so digits
    arrive reversed. An FST cannot reorder without enumerating, so this enumerates every
    ones/tens pair present in the digit and tens tables.
    """
    ones_digits = sorted({output for _, output, _ in digits.paths().items()})
    tens_digits = sorted({output for _, output, _ in tens.paths().items()})
    return pynini.string_map([(one + ten, ten + one) for one in ones_digits for ten in tens_digits])


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus dreiundzwanzig -> cardinal { negative: "true" integer: "23" }
        e.g. eintausend -> cardinal { integer: "1.000" }
    Numbers below thirteen are not converted.
    The transducer implements a period separator every three digits by default.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # WFST mappings for numbers 0-99
        zero = pynini.string_file(get_abs_path("data/cardinal/zero.tsv"))
        digits = pynini.string_file(get_abs_path("data/cardinal/digits.tsv"))
        # Isolates single digit cardinals to pass to other graphs
        self.digits = digits.optimize()
        irregular_teens = pynini.string_file(get_abs_path("data/cardinal/irregular_teens.tsv"))
        to_denormalize = zero | digits | irregular_teens

        # Isolates the first dozen
        self.dozen = to_denormalize.optimize()
        teens = pynini.string_file(get_abs_path("data/cardinal/teens.tsv"))
        tens = pynini.string_file(get_abs_path("data/cardinal/tens.tsv"))
        ties = tens + pynutil.insert("0")
        # German flips ones and tens in two-digit numbers. The WFST below handles these flips.
        delete_space = pynutil.delete(NEMO_SPACE)
        delete_und = pynutil.delete("und")

        # Accepts normalized digits+ties (ein+und+zwanzig)
        digit_ties = digits + delete_space.ques + delete_und + delete_space.ques + tens
        # Flips ties and digits for denormalization
        ties_digit = digit_ties @ swap_tens_and_ones(digits, tens)

        # WFST grammar for hundreds
        graph_10_99 = irregular_teens | teens | ties | ties_digit
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
            | ((digits | pynutil.insert("1")) + delete_space.ques + pynini.cross("hundert", "00"))
        )

        # Digits are grouped in clusters of three: {hundreds}{tens}{ones}.
        # Clusters of three are separated by periods, applied right to left.
        non_zero_digit_cluster = (hundreds) | (pynutil.insert("0") + graph_10_99) | (pynutil.insert("00") + digits)
        digit_cluster = non_zero_digit_cluster | pynutil.insert("000")

        # WFST grammar for thousands
        thousands = (pynini.cross("tausend", "1.000")) | (
            (
                (pynini.cross("tausend", "1.") + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross("tausend", ".") + delete_und.ques)
                | pynutil.insert("000.")
            )
            + delete_space.ques
            + digit_cluster
        )

        # WFST grammar for millions
        million = pynini.accep("million") | pynini.accep("millionen")
        millions = (pynini.cross("million", "1.000.000")) | (
            (
                (pynini.cross("million", "1.") + delete_space.ques + delete_und.ques)
                | (digit_cluster + delete_space.ques + pynini.cross(million, ".") + delete_und.ques)
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
                    pynini.cross((pynini.accep("milliarde") | pynini.accep("milliard")), "1.")
                    + delete_space.ques
                    + delete_und.ques
                )
                | (digit_cluster + delete_space.ques + pynini.cross(billion, ".") + delete_und.ques)
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
                | (digit_cluster + delete_space.ques + pynini.cross(trillion, ".") + delete_und.ques)
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
                | (digit_cluster + delete_space.ques + pynini.cross(quadrillion, ".") + delete_und.ques)
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
                | (digit_cluster + delete_space.ques + pynini.cross(quintillion, ".") + delete_und.ques)
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
                | (digit_cluster + delete_space.ques + pynini.cross(sextillion, ".") + delete_und.ques)
                | pynutil.insert("000.")
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

        # the name the other German semiotic classes use for the graph without the first-dozen exception
        self.graph_no_exception = self.graph_all_cardinals

        # 1-999 without leading zeros, consumed by the decimal tagger's get_quantity
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            non_zero_digit_cluster @ remove_leading_zeros
        ).optimize()

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
            pynutil.insert("negative: ") + pynini.cross("minus ", '"true"') + pynutil.insert(" "),
            0,
            1,
        )

        # the decimal verbalizer reads a single character out of the negative field, so the graph
        # handed to the other classes keeps the minus sign rather than the "true" flag
        self.optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus ", '"true"') + pynutil.insert(" "), 0, 1
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
