# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.de.taggers.cardinal import CARDINAL_SCALES
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


def get_quantity(decimal: 'pynini.FstLike', cardinal: GraphFst, deterministic: bool = True) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or a decimal followed by a quantity into a numeral
        e.g. zehn millionen -> integer_part: "10" quantity: "Mio."
        e.g. zehn komma fünf millionen -> integer_part: "10" fractional_part: "5" quantity: "Mio."

    Args:
        decimal: decimal FST
        cardinal: CardinalFst, provides the shared magnitude graph and the 1 - 999 graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated
    """
    quantity = cardinal.magnitude
    if not deterministic:
        quantity |= pynutil.add_weight(
            pynini.string_file(get_abs_path("data/numbers/quantity_nondeterministic.tsv")), 0.001
        )

    # after a bare integer "hundert" and "tausend" belong to the cardinal grammar: zwei hundert -> 200
    cardinal_words = pynini.union(*[cardinal.scale_forms[scale] for scale in CARDINAL_SCALES]).optimize()
    big_quantity = pynini.compose(pynini.difference(cardinal.magnitude_words, cardinal_words).optimize(), quantity)

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
    The tagger accepts canonical verbalized decimal input whereby every digit after the comma is pronounced separately:
        e.g. 12,345 -> zwölf komma drei vier fünf
            *12,345 -> zwölf komma drei hundert fünfundvierzig
    Tausend, million and milliarde are denormalized to their abbreviated forms, hundert and larger magnitudes
    keep the full word:
        e.g. tausend -> Tsd.
             million -> Mio.
             milliarde -> Mrd.
             billionen -> Billionen
    A bare integer followed by "hundert" or "tausend" stays with the cardinal grammar, so the abbreviation only
    shows up after a decimal: dreiviertel tausend -> 0,75 Tsd. but zwei tausend -> 2.000
    For deterministic=False the full word forms of tausend, million and milliarde are generated as well:
        e.g. millionen -> Mio. | Millionen

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

        # Handles cases where the integer may be missing before the comma and inserts a '0' in its place
        graph_integer_or_zero = graph_integer | pynutil.insert('integer_part: "0" ', weight=-0.001)

        graph_clean_digit = delete_space + graph_digit

        # Digits post-comma are pronounced individually
        graph_string_of_digits = pynini.closure(graph_clean_digit, 1)
        graph_fractional = pynutil.insert('fractional_part: "') + graph_string_of_digits + pynutil.insert('"')

        graph_decimal_no_sign = graph_integer_or_zero + delete_comma + graph_fractional

        # Coverage for verbalized 0,5 (einhalb)
        half = pynini.cross("einhalb", 'fractional_part: "5"')
        einhalb = graph_integer_or_zero + half

        graph_decimal_no_sign |= einhalb

        # Coverage for verbalized 1,5 (andterthald, einanderthalb)
        one_and_a_half = pynini.accep("anderthalb") | pynini.accep("einanderthalb")
        graph_halves = pynini.cross(one_and_a_half, 'integer_part: "1" fractional_part: "5"')

        graph_decimal_no_sign |= graph_halves

        # Coverage for verbalized 0,25 (einviertel) and 0,75 (dreiviertel)
        graph_quarters = pynini.string_map(
            [
                ("einviertel", 'integer_part: "0" fractional_part: "25"'),
                ("dreiviertel", 'integer_part: "0" fractional_part: "75"'),
            ]
        )

        graph_decimal_no_sign |= graph_quarters

        # measure and money splice this in and write the sign themselves
        self.final_graph_wo_negative = (
            graph_decimal_no_sign | get_quantity(graph_decimal_no_sign, cardinal, deterministic=deterministic)
        ).optimize()

        final_graph = cardinal.optional_minus_graph + self.final_graph_wo_negative
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
