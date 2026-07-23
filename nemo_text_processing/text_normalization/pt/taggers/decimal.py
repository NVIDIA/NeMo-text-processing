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

from nemo_text_processing.text_normalization.pt.graph_utils import NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.pt.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese decimal numbers, e.g.
        "1,26" -> decimal { integer_part: "um" fractional_part: "dois seis" }
        "0,01" -> decimal { integer_part: "zero" fractional_part: "zero um" }
        "-1,26" -> decimal { negative: "true" ... }
        "1,33 milhões" / "1 milhão" -> decimal { ... quantity: "milhões" / "milhão" }

    The fractional mantissa (after the comma) is always read digit-by-digit (0–9),
    including leading zeros. Integer part and quantities still use cardinals.

    Args:
        cardinal: CardinalFst instance for integer verbalization in tags.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        _num = lambda name: pynini.string_file(get_abs_path(f"data/numbers/{name}")).optimize()

        comma = pynutil.delete(",")
        quantity_words = _num("quantity_words.tsv")
        digit = _num("digit.tsv")
        zero = _num("zero.tsv")
        graph_digit_or_zero = pynini.union(digit, zero)
        digit_by_digit = (graph_digit_or_zero + pynini.closure(insert_space + graph_digit_or_zero)).optimize()

        fractional_digits = pynini.closure(NEMO_DIGIT, 1, 15)
        graph_fractional = (
            pynutil.insert('fractional_part: "') + (fractional_digits @ digit_by_digit) + pynutil.insert('"')
        )

        non_zero_lead = pynini.difference(NEMO_DIGIT, pynini.accep("0"))

        graph_integer_zero = (
            pynutil.insert('integer_part: "') + pynini.cross("0", "zero") + pynutil.insert('"') + insert_space
        )
        decimal_when_zero = graph_integer_zero + comma + insert_space + graph_fractional

        graph_integer_pos = (
            pynutil.insert('integer_part: "')
            + (non_zero_lead + pynini.closure(NEMO_DIGIT, 0, 11)) @ cardinal_graph
            + pynutil.insert('"')
            + insert_space
        )
        decimal_when_pos = graph_integer_pos + comma + insert_space + graph_fractional

        decimal_core = pynini.union(decimal_when_zero, decimal_when_pos)
        integer_quantity = (
            pynutil.insert('integer_part: "')
            + (pynini.closure(NEMO_DIGIT, 1, 12) @ cardinal_graph)
            + pynutil.insert('"')
            + insert_space
            + delete_space
            + pynutil.insert('quantity: "')
            + quantity_words
            + pynutil.insert('"')
        )
        decimal_quantity = (
            decimal_core + delete_space + pynutil.insert('quantity: "') + quantity_words + pynutil.insert('"')
        )
        final_graph_wo_sign = pynini.union(decimal_core, integer_quantity, decimal_quantity)
        self.final_graph_wo_negative = final_graph_wo_sign.optimize()
        optional_minus = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)
        final_graph = optional_minus + final_graph_wo_sign

        self.fst = self.add_tokens(final_graph).optimize()
