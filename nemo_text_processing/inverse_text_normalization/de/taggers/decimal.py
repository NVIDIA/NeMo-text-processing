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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


def get_quantity(decimal: 'pynini.FstLike', cardinal: GraphFst, deterministic: bool = True) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or a decimal followed by a quantity into a numeral
        e.g. zehn millionen -> integer_part: "10" quantity: "Millionen"
        e.g. zehn komma fünf millionen -> integer_part: "10" fractional_part: "5" quantity: "Millionen"

    Args:
        decimal: decimal FST
        cardinal: CardinalFst, provides the shared magnitude graph and the 1 - 999 graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated
    """
    quantity = cardinal.magnitude
    if not deterministic:
        # Tsd., Mio. and Mrd. are the only magnitudes German abbreviates, a convention
        quantity |= pynini.string_file(get_abs_path("data/numbers/quantity_nondeterministic.tsv"))

    big_quantity = pynini.compose(cardinal.big_magnitude_words, quantity)

    res = (
        pynutil.insert('integer_part: "')
        + cardinal.graph_hundred_component_at_least_one_none_zero_digit
        + pynutil.insert('"')
        + pynutil.insert(' quantity: "')
        + delete_space
        + big_quantity
        + pynutil.insert('"')
    )
    res |= decimal + pynutil.insert(' quantity: "') + delete_space + quantity + pynutil.insert('"')
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers
        e.g. minus elf komma zwei null null sechs billionen -> decimal { negative: "-" integer_part: "11"  fractional_part: "2006" quantity: "Billionen" }
    The tagger accepts canonical verbalized decimal input whereby every digit after the comma is pronounced
    separately, so zwölf komma drei hundert fünfundvierzig is not read as a single decimal:
        e.g. zwölf komma drei vier fünf -> decimal { integer_part: "12" fractional_part: "345" }
    Magnitude words keep their full written form, the nouns capitalised and the numeral "tausend" lower case.
    After a bare integer "hundert" and "tausend" stay with the cardinal grammar, the larger scales do not:
        e.g. eine million -> decimal { integer_part: "1" quantity: "Million" }
        e.g. dreiviertel tausend -> decimal { integer_part: "0" fractional_part: "75" quantity: "tausend" }
    Only tausend, million and milliarde have abbreviated forms:
        for non-deterministic case: eine million ->
            decimal { integer_part: "1" quantity: "Mio." }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)
        graph_cardinals = cardinal.graph_no_exception
        delete_comma = pynutil.delete("komma")
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digits.tsv"))
        graph_digit |= pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        graph_integer = pynutil.insert('integer_part: "') + graph_cardinals + pynutil.insert('" ') + delete_space

        graph_integer_or_zero = graph_integer | pynutil.insert('integer_part: "0" ', weight=-0.001)

        graph_clean_digit = delete_space + graph_digit

        graph_string_of_digits = pynini.closure(graph_clean_digit, 1)
        graph_fractional = pynutil.insert('fractional_part: "') + graph_string_of_digits + pynutil.insert('"')

        graph_decimal_no_sign = graph_integer_or_zero + delete_comma + graph_fractional

        fraction_values = pynini.string_file(get_abs_path("data/numbers/fractions.tsv"))

        value_to_fields = (
            pynutil.insert('integer_part: "')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynini.cross(",", '" fractional_part: "')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.insert('"')
        )
        # "einhalb" attaches to a preceding integer (zwei einhalb -> 2,5), so it contributes only
        # the digits after the comma and the integer part comes from graph_integer_or_zero
        value_to_fractional = (
            pynutil.insert('fractional_part: "')
            + pynutil.delete(pynini.closure(NEMO_DIGIT, 1) + ",")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.insert('"')
        )

        half = pynini.compose(pynini.accep("einhalb") @ fraction_values, value_to_fractional)
        graph_decimal_no_sign |= graph_integer_or_zero + half

        standalone = pynini.difference(pynini.project(fraction_values, "input"), pynini.accep("einhalb"))
        graph_decimal_no_sign |= pynini.compose(standalone @ fraction_values, value_to_fields)

        # measure and money splice this in and write the sign themselves
        self.final_graph_wo_negative = (
            graph_decimal_no_sign | get_quantity(graph_decimal_no_sign, cardinal, deterministic=deterministic)
        ).optimize()

        final_graph = cardinal.optional_minus_graph + self.final_graph_wo_negative
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
