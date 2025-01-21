# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst, insert_space
from nemo_text_processing.text_normalization.ar.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
    321.7 --> ثلاث مئة وواحد وعشرون وسبعة من عشرة
    -321.7  -> decimal { negative: "true" integer_part: "321"  fractional_part: ".7" }
    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        integer_part = cardinal.cardinal_numbers
        cardinal_numbers_with_leading_zeros = cardinal.cardinal_numbers_with_leading_zeros
        self.integer_part = pynini.closure(integer_part, 0, 1)
        self.seperator = pynini.string_map([(".", "و"), (",", "و")])

        add_preposition = pynutil.insert(" من ")
        graph_fractional = NEMO_DIGIT @ cardinal_numbers_with_leading_zeros + add_preposition + pynutil.insert("عشرة")
        graph_fractional |= (
            (NEMO_DIGIT + NEMO_DIGIT) @ cardinal_numbers_with_leading_zeros + add_preposition + pynutil.insert("مئة")
        )
        graph_fractional |= (
            (NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT) @ cardinal_numbers_with_leading_zeros
            + add_preposition
            + pynutil.insert("ألف")
        )
        graph_fractional |= (
            (NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT) @ cardinal_numbers_with_leading_zeros
            + add_preposition
            + pynutil.insert("عشرة آلاف")
        )

        graph_integer = pynutil.insert('integer_part: "') + self.integer_part + pynutil.insert('" ')
        # to parse something like ,50 alone as well
        graph_integer_or_none = graph_integer | pynutil.insert('integer_part: "0" ', weight=0.001)

        self.optional_quantity = pynini.string_file(get_abs_path("data/number/quantities.tsv")).optimize()
        self.graph_fractional = graph_fractional

        graph_fractional = (
            pynutil.insert('fractional_part: "') + self.seperator + graph_fractional + pynutil.insert('"')
        )
        optional_quantity = pynini.closure(
            (pynutil.add_weight(pynini.accep(NEMO_SPACE), -0.1) | insert_space)
            + pynutil.insert('quantity: "')
            + self.optional_quantity
            + pynutil.insert('"'),
            0,
            1,
        )

        self.graph_decimal = self.integer_part + insert_space + self.seperator + graph_fractional

        self.final_graph_decimal = (
            cardinal.optional_minus_graph + graph_integer_or_none + insert_space + graph_fractional + optional_quantity
        )

        self.final_graph = self.add_tokens(self.final_graph_decimal)
        self.fst = self.final_graph.optimize()
