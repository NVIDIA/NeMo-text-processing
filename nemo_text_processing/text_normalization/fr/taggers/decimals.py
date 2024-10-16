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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.fr.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))


def get_quantity(decimal_graph: "pynini.FstLike", cardinal_graph: "pynini.FstLike") -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 2 millions -> integer_part: "deux" quantity: "millions"
    e.g. 2,4 millions -> integer_part: "deux" fractional_part: "quatre" quantity: "millions"
    Args:
        decimal_graph: DecimalFST
        cardinal_graph: CardinalFST
    """
    delete_separator = pynini.closure(pynutil.delete(" "), 0, 1)
    numbers = pynini.closure(NEMO_DIGIT, 1, 6) @ cardinal_graph
    numbers = pynini.cdrewrite(pynutil.delete(delete_separator), "", "", NEMO_SIGMA) @ numbers

    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + NEMO_SPACE
        + pynutil.insert('quantity: "')
        + quantities
        + pynutil.insert('"')
    )
    res |= decimal_graph + NEMO_SPACE + pynutil.insert('quantity: "') + quantities + pynutil.insert('"')
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -11,406 millions -> decimal { negative: "true" integer_part: "onze" fractional_part: "quatre cent six" quantity: "millions" preserve_order: true }
        114 billions -> decimal { integer_part: "cent quatorze" quantity: "billions" preserve_order: true }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = digit | zero

        if not deterministic:
            graph = pynini.union(graph_digit, cardinal.all_hundreds, cardinal.one_to_all_tens)
            graph += pynini.closure(insert_space + graph)

        else:
            # General pattern is 1-3 digits: map as cardinal, default to tens followed by digits otherwise \
            graph = pynini.union(
                pynutil.add_weight(graph_digit + pynini.closure(insert_space + zero), -0.00001),
                pynutil.add_weight(cardinal.all_double_digits + pynini.closure(insert_space + zero), -0.00002),
                pynutil.add_weight(cardinal.all_hundreds + pynini.closure(insert_space + zero), 0.00001),
                pynutil.add_weight(
                    cardinal.all_double_digits
                    + pynini.closure(insert_space + cardinal.all_double_digits, 1)
                    + pynini.closure(insert_space + zero, 0, 1)
                    + (pynini.closure(insert_space + graph_digit, 0, 1) | pynini.closure(insert_space + zero, 0)),
                    -0.00002,
                ),  # Read out as tens and a possible trailing digit or zeroes
                zero
                + pynini.closure(insert_space + zero)
                + pynini.closure(insert_space + graph_digit),  # For cases such as "1,001"
            )

        graph = graph.optimize()

        delete_separator = pynutil.delete(",")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        graph_fractional = pynutil.insert('fractional_part: "') + graph + pynutil.insert('"')

        integers = cardinal.all_nums_no_tokens
        graph_integer = pynutil.insert('integer_part: "') + integers + pynutil.insert('"')
        final_graph_wo_sign = graph_integer + delete_separator + insert_space + graph_fractional

        final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, integers).optimize()
        self.final_graph_wo_negative = final_graph_wo_negative

        final_graph = optional_graph_negative + final_graph_wo_negative
        self.final_graph = final_graph

        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
