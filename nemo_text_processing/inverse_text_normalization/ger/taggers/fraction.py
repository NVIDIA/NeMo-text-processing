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
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions
        e.g. ein halb -> fraction { numerator: "1" denominator: "2" }
        e.g. ein drei viertel -> fraction { integer_part: "1" numerator: "3" denominator: "4" }
        e.g. minus vier fünfhundertzwei eintausendeinhundertzehntel -> fraction { negative: "-" integer_part: "4" numerator: "502" denominator: "1110" }
    Even powers of 10 are denormalized to their abbreviated forms:
        e.g. million -> Mio.
             millard -> Mrd.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        graph_cardinals = cardinal.graph_all_cardinals
        accep_space = pynini.accep(NEMO_SPACE)

        # Logic for the denominator
        # Defines an acceptor for denominator suffixes: -stel, -steln, -tel, -teln
        suffix = pynini.accep("s").ques + pynini.accep("tel") + pynini.accep("n").ques

        # Strips denominator suffixes from cardinal numerals with syllabically reduced stems
        exceptions = (
            pynini.cross(("drit" + suffix), "drei")
            | pynini.cross(("sech" + suffix), "sechs")
            | pynini.cross(("sieb" + suffix), "sieben")
            | pynini.cross(("ach" + suffix), "acht")
        )

        # Accounts for the suppletive "halbe" for "2" in the denominator
        exceptions |= pynini.cross("halbe", "zwei")

        # Defines an acceptor for all cardinal numerals
        cardinals_acceptor = pynini.project(graph_cardinals, "input")
        # Removes the numerals with stems undergoing reduction
        irregular_cardinals = pynini.union("drei", "sechs", "sieben", "acht")
        cardinals_acceptor = pynini.difference(cardinals_acceptor, irregular_cardinals)
        # Defines an acceptor for all regularly derived denominators
        regulars = cardinals_acceptor + suffix

        # Removes suffixes
        remove_suffix = pynini.cdrewrite(
            pynutil.delete(suffix), "", "[EOS]", NEMO_SIGMA
        )

        # Graphs the denominator
        # Transduces from inflected numerals to numbers
        graph_denominator_numbers = exceptions | (regulars @ remove_suffix)
        graph_denominator = graph_denominator_numbers @ graph_cardinals

        # Logic for the nominal morphology of "1" in the integer place
        morphemes_for_one = (
            pynini.accep("ein") + pynini.accep("e").ques + pynini.union(*"rnsm").ques
        )

        graph_one = pynini.cross(morphemes_for_one, "1")

        # Logic for the whole fraction
        # Since period separators aren't permitted in fractions, they have to be removed
        remove_period_separators = pynini.cdrewrite(
            pynutil.delete("."), "", "", NEMO_SIGMA
        )
        numerator_value = graph_cardinals @ remove_period_separators 
        self.cardinals_no_separators = numerator_value# provides a graph for digits without period separators
        denominator_value = graph_denominator @ remove_period_separators
        integer_part = (
            pynutil.insert('integer_part: "')
            + (graph_cardinals | graph_one)
            + pynutil.insert('"')
        )
        numerator_part = (
            pynutil.insert('numerator: "') + numerator_value + pynutil.insert('"')
        )
        denominator_part = (
            pynutil.insert('denominator: "') + denominator_value + pynutil.insert('"')
        )

        # Graphs the fraction in absolute value
        graph_fraction_ABS = (
            integer_part + accep_space + numerator_part + accep_space + denominator_part
        ) | (numerator_part + accep_space + denominator_part)

        graph_fraction_raw = (
            (graph_cardinals | graph_one)
            + accep_space
            + numerator_value
            + pynini.cross(" ", "/")
            + denominator_value
        ) | (numerator_value + pynini.cross(" ", "/") + denominator_value)

        # Logic to handle one-half (e.g. "ein halb")
        # Note: "einhalb" and "anderthalb" are implemented as "0,5" and "1,5" respectively in the DECIMAL class
        # Note: "einviertel" and "dreiviertel" are implemented as "0,25" and "0,75" respectively in the DECIMAL class
        half_numerator_part = (
            pynutil.insert('numerator: "') + graph_one + pynutil.insert('"')
        )

        half_denominator_suffix = pynini.accep("e") + pynini.union(*"rnsm").ques
        half = pynini.accep("halb") + half_denominator_suffix.ques
        half_denominator = pynini.cross(half, "2")
        half_denominator_part = (
            pynutil.insert('denominator: "') + half_denominator + pynutil.insert('"')
        )

        graph_half_ABS = (
            integer_part
            + accep_space
            + half_numerator_part
            + accep_space
            + half_denominator_part
        ) | (half_numerator_part + accep_space + half_denominator_part)

        graph_fraction_ABS |= graph_half_ABS

        graph_half_ABS_raw = (
            (graph_cardinals | graph_one)
            + accep_space
            + graph_one
            + pynini.cross(" ", "/")
            + half_denominator
        ) | (graph_one + pynini.cross(" ", "/") + half_denominator)

        graph_fraction_raw |= graph_half_ABS_raw

        # Logic to handle fractions with missing numerators
        # e.g. viertel -> 1/4
        missing_numerator = pynutil.insert('numerator: "1"')
        all_denominators = half_denominator | denominator_value
        graph_stranded_denominator = (
            pynutil.insert('denominator: "') + all_denominators + pynutil.insert('"')
        )
        graph_missing_numerator_ABS = (
            missing_numerator + pynutil.insert(NEMO_SPACE) + graph_stranded_denominator
        )

        graph_fraction_ABS |= graph_missing_numerator_ABS

        graph_missing_numerator_abs_raw = (
            pynutil.insert("1")
            + pynini.cross(" ", "/")
            + (half_denominator | denominator_value)
        )

        graph_fraction_raw |= graph_missing_numerator_abs_raw

        # Utilizes the "morphosyntactic_features" field to handle even powers of ten (eg. Million, Billion, etc.)
        quantity = pynini.string_file(get_abs_path("/data/fraction/quantity.tsv"))
        graph_quantity = (
            pynutil.delete(NEMO_SPACE).ques
            + pynutil.insert(' morphosyntactic_features: "')
            + quantity
            + pynutil.insert('"')
        )

        graph_fraction_ABS += graph_quantity.ques

        graph_quantity_raw = pynini.accep(NEMO_SPACE) + quantity

        graph_fraction_raw += graph_quantity_raw.ques
        self.graph_fraction_raw = graph_fraction_raw

        self.graph_fraction_ABS = graph_fraction_ABS

        # Handles the negative sign
        graph_negative = pynini.cross("minus", 'negative: "-"') + accep_space
        graph_fraction = graph_negative.ques + graph_fraction_ABS
        self.graph_fraction = graph_fraction

        # Generates the final graph
        graph = self.add_tokens(graph_fraction)
        self.fst = graph.optimize()
