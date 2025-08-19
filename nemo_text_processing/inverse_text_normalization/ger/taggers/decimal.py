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
    delete_space,
    NEMO_SPACE,
    GraphFst,
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers
        e.g. minus elf komma zwei null null sechs billionen -> decimal { negative: "-" integer_part: "11"  fractional_part: "2006" quantity: "Bio." }
    The tagger accepts canonical verbalized decimal input whereby every digit after the comma is pronounced separately:
        e.g. 12,345 -> zwölf komma drei vier fünf
            *12,345 -> zwölf komma drei hundert fünfundvierzig
    Even powers of 10 are denormalized to their abbreviated forms:
        e.g. million -> Mio.
             millard -> Mrd.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")
        graph_cardinals = cardinal.graph_all_cardinals
        delete_comma = pynutil.delete("komma")
        graph_digit = pynini.string_file(get_abs_path("data/decimal/digits.tsv"))

        graph_integer = (
            pynutil.insert('integer_part: "')
            + graph_cardinals
            + pynutil.insert('" ')
            + delete_space
        )

        # Handles cases where the integer may be missing before the comma and inserts a '0' in its place
        graph_integer_or_zero = graph_integer | pynutil.insert(
            'integer_part: "0" ', weight=-0.001
        )

        graph_clean_digit = delete_space + graph_digit

        # Digits post-comma are pronounced individually
        graph_string_of_digits = pynini.closure(graph_clean_digit, 1)
        graph_fractional = (
            pynutil.insert('fractional_part: "')
            + graph_string_of_digits
            + pynutil.insert('"')
        )

        graph_decimal_no_sign = graph_integer_or_zero + delete_comma + graph_fractional

        # Coverage for verbalized 0,5 (einhalb)
        half = pynini.cross("einhalb", 'fractional_part: "5"')
        einhalb = graph_integer_or_zero + delete_space.ques + half

        graph_decimal_no_sign |= einhalb

        # Coverage for verbalized 1,5 (andterthald, einanderthalb)
        one_and_a_half = pynini.accep("anderthalb") | pynini.accep("einanderthalb")
        graph_halves = pynini.cross(
            one_and_a_half, 'integer_part: "1" fractional_part: "5"'
        )

        graph_decimal_no_sign |= graph_halves

        # Coverage for verbalized 0,25 (einviertel) and 0,75 (dreiviertel)
        graph_quarters = pynini.string_map(
            [
                ("einviertel", 'integer_part: "0" fractional_part: "25"'),
                ("dreiviertel", 'integer_part: "0" fractional_part: "75"'),
            ]
        )

        graph_decimal_no_sign |= graph_quarters

        # Handles the negative sign
        graph_negative = pynini.cross("minus", 'negative: "-" ') + delete_space

        graph_decimal = graph_negative.ques + graph_decimal_no_sign

        # Utilizes the quantity field to handle even powers of ten (eg. Million, Billion, etc.)
        quantity = pynini.string_file(get_abs_path("data/decimal/quantity.tsv"))
        graph_quantity = (
            pynutil.insert(' quantity: "')
            + delete_space.ques
            + quantity
            + pynutil.insert('"')
        )

        graph_decimal += graph_quantity.ques
        self.graph_decimal = graph_decimal
        graph = self.add_tokens(graph_decimal)
        self.fst = graph.optimize()
