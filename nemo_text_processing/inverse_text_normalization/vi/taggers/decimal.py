# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. âm hai hai phẩy không năm tư năm tỉ -> decimal { negative: "true" integer_part: "22"  fractional_part: "054" quantity: "tỉ" }
        e.g. không chấm ba lăm -> decimal { integer_part: "0" fractional_part: "35" }
        e.g. một triệu rưỡi -> decimal { integer_part: "1" quantity: "triệu" fractional_part: "5" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        base_decimal = graph_digit | graph_zero
        graph_four = pynini.cross("tư", "4")
        graph_five = pynini.cross("lăm", "5")
        graph_one = pynini.cross("mốt", "1")
        negative_words = pynini.union("âm", "trừ")

        graph_decimal = pynini.union(
            base_decimal,
            graph_four,
            pynini.closure(base_decimal + delete_space, 1) + (base_decimal | graph_four | graph_five | graph_one),
        ).optimize()
        self.graph = graph_decimal

        point = pynutil.delete("chấm") | pynutil.delete("phẩy")
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative:") + insert_space + pynini.cross(negative_words, '"true"') + delete_extra_space,
            0,
            1,
        )

        graph_fractional = (
            pynutil.insert('fractional_part:')
            + insert_space
            + pynutil.insert('"')
            + graph_decimal
            + pynutil.insert('"')
        )
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )
        # Build quantity handling - reuse magnitude words from cardinal context
        # e.g. một triệu -> integer_part: "1" quantity: "triệu"
        # e.g. một tỷ rưỡi -> integer_part: "1" fractional_part: "5" quantity: "tỷ"
        numbers = cardinal.graph_hundred_component_at_least_one_none_zero_digit @ (
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
        )

        magnitude_words = pynini.union("triệu", "tỉ", "tỷ", "vạn")
        thousand_words = pynini.union("ngàn", "nghìn")

        last_digit = cardinal.last_digit
        optional_fraction_graph = pynini.closure(
            delete_extra_space
            + pynutil.insert('fractional_part:')
            + insert_space
            + pynutil.insert('"')
            + (last_digit | pynini.cross("rưỡi", "5") | graph_one | graph_four)
            + pynutil.insert('"'),
            0,
            1,
        )

        quantity_graph = (
            pynutil.insert('integer_part:')
            + insert_space
            + pynutil.insert('"')
            + numbers
            + pynutil.insert('"')
            + delete_extra_space
            + pynutil.insert('quantity:')
            + insert_space
            + pynutil.insert('"')
            + magnitude_words
            + pynutil.insert('"')
            + optional_fraction_graph
        )
        quantity_graph |= (
            final_graph_wo_sign
            + delete_extra_space
            + pynutil.insert('quantity:')
            + insert_space
            + pynutil.insert('"')
            + (magnitude_words | thousand_words)
            + pynutil.insert('"')
        )

        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | quantity_graph
        final_graph |= optional_graph_negative + quantity_graph
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
