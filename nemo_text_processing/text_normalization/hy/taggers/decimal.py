# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.hy.utils import get_abs_path


def get_quantity(decimal_graph: "pynini.FstLike", cardinal_graph: "pynini.FstLike") -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 2 միլիոն -> integer_part: "երկու" quantity: "միլիոն"
    e.g. 2․4 միլիոն -> integer_part: "երկու" fractional_part: "չորս" quantity: "միլիոն"
    Args:
        decimal_graph: DecimalFST
        cardinal_graph: CardinalFST
    """
    quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
    delete_separator = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1)
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
        554 միլիարդ -> decimal { integer_part: "հինգ հարյուր հիսունչորս" quantity: "միլիարդ" }
    Args:
        cardinal: CardinalFst
        deterministic is not necessary right now
        TODO make deterministic make sense
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph = cardinal.one_to_all_tens

        graph = graph.optimize()

        delete_separator = pynutil.delete(".") | pynutil.delete("․")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        graph_fractional = pynutil.insert('fractional_part: "') + graph + pynutil.insert('"')

        integers = cardinal.all_nums_no_tokens
        graph_integer = pynutil.insert('integer_part: "') + integers + pynutil.insert('"')
        final_graph_wo_sign = graph_integer + delete_separator + insert_space + graph_fractional

        final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, integers)
        self.final_graph_wo_negative = final_graph_wo_negative.optimize()

        final_graph = optional_graph_negative + final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
