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
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. trừ hai mươi ba -> cardinal { integer: "23" negative: "-" } }
        e.g. hai nghìn lẻ chín -> cardinal { integer: "2009"} }
    Numbers below ten are not converted.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        thousand_words = pynini.union("ngàn", "nghìn")
        negative_words = pynini.union("âm", "trừ")

        graph_hundred = pynini.cross("trăm", "")
        graph_ten = pynini.cross("mươi", "")
        zero = pynini.cross(pynini.union("linh", "lẻ"), "0")

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_one = pynini.cross("mốt", "1")
        graph_four = pynini.cross("tư", "4")
        graph_five = pynini.cross("lăm", "5")
        graph_half = pynini.cross("rưỡi", "5")

        optional_ten = pynini.closure(delete_space + graph_ten, 0, 1)
        last_digit_exception = pynini.project(pynini.cross("năm", "5"), "input")
        self.last_digit = pynini.union(
            (pynini.project(graph_digit, "input") - last_digit_exception.arcsort()) @ graph_digit,
            graph_one,
            graph_four,
            graph_five,
        )
        last_digit = self.last_digit
        # Build hundreds component (e.g., "một trăm", "hai trăm")
        graph_hundreds_component = (graph_digit | graph_zero) + delete_space + graph_hundred
        graph_hundreds_component += delete_space
        graph_hundreds_component += pynini.union(
            graph_teen,
            (graph_half | graph_four | graph_one) + pynutil.insert("0", weight=0.1),
            graph_ties + optional_ten + ((delete_space + last_digit) | pynutil.insert("0", weight=0.1)),
            zero + delete_space + (graph_digit | graph_four | graph_five),
            pynutil.insert("00", weight=0.1),
        ).optimize()
        graph_hundreds_component |= (
            pynutil.insert("0")
            + delete_space
            + pynini.union(
                graph_teen,
                graph_ties + optional_ten + delete_space + last_digit,
                graph_ties + delete_space + graph_ten + pynutil.insert("0", weight=0.1),
                zero + delete_space + (graph_digit | graph_four | graph_five),
            ).optimize()
        )
        graph_hundred_component = graph_hundreds_component | (
            pynutil.insert("00", weight=0.1) + delete_space + graph_digit
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit.optimize()
        )
        graph_hundreds_zero = graph_hundreds_component | pynutil.insert("000", weight=0.1)

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete(thousand_words),
            pynutil.insert("000", weight=0.1),
        ).optimize()

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("triệu"),
            pynutil.insert("000", weight=0.1),
        ).optimize()
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete(pynini.union("tỉ", "tỷ")),
            pynutil.insert("000", weight=0.1),
        ).optimize()

        # Main graph combining all magnitude levels
        graph = pynini.union(
            # Full format: billion + million + thousand + hundred
            graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundreds_zero,
            # Special thousand format with last digit or "rưỡi" (half)
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete(thousand_words)
            + delete_space
            + (((last_digit | graph_half) + pynutil.insert("00", weight=0.1)) | graph_hundreds_zero),
            # Single digits (for non-exception cases)
            graph_digit,
            graph_zero,
        )

        graph = (
            graph
            @ pynini.union(
                pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT),
                "0",
            ).optimize()
        )

        # don't convert cardinals from zero to nine inclusive
        single_digits = pynini.project(pynini.union(graph_digit, graph_zero), "input").optimize()

        self.graph_no_exception = graph

        self.graph = pynini.difference(pynini.project(graph, "input"), single_digits) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(negative_words, '"-"') + NEMO_SPACE,
            0,
            1,
        )

        final_graph = optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
