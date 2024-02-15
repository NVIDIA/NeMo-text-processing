# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.hy.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. իննսունյոթ -> cardinal { integer: "97" } }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        zero = pynini.string_map([("զրո", "0")])
        digit = (pynini.string_file(get_abs_path("data/numbers/digit.tsv"))) + (
            pynini.closure(pynutil.delete("ն") | pynutil.delete("ի") | pynutil.delete("ին"), 0, 1)
        )
        digits_no_one = pynini.string_file(get_abs_path("data/numbers/digits_no_one.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv")) + (
            pynini.closure(pynutil.delete("ն") | pynutil.delete("ի") | pynutil.delete("ին"), 0, 1)
        )
        graph_digit = digit | pynutil.insert("0")

        graph_ties = graph_ties | pynutil.insert("0")
        graph_two_digit_nums = graph_ties + graph_digit

        hundred = pynini.accep("հարյուր")
        graph_hundred = pynini.cross("հարյուր", "1")

        graph_hundreds_first_digit = graph_hundred | (digits_no_one + delete_space + pynutil.delete(hundred))
        graph_hundreds = (
            (graph_hundreds_first_digit + delete_space | pynutil.insert("0", weight=0.1))
            + delete_space
            + graph_two_digit_nums
        )

        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundreds @ (pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)).optimize()
        )

        graph_one_thousand = pynini.cross("հազար", "1")
        graph_many_thousand = graph_hundreds + delete_space + pynutil.delete("հազար")
        graph_thousands = (
            (graph_one_thousand | graph_many_thousand | pynutil.insert("000", weight=0.000000001))
            + delete_space
            + graph_hundreds
        )

        millions = pynini.accep("միլիոն")
        graph_millions = (
            ((graph_hundreds + delete_space + pynutil.delete(millions)) | pynutil.insert("000", weight=0.1))
            + delete_space
            + graph_thousands
        )

        billions = pynini.accep("միլիարդ")
        graph_billions = (
            (graph_hundreds + delete_space + pynutil.delete(billions) + delete_space)
            | pynutil.insert("000", weight=0.1)
        ) + graph_millions

        trillions = pynini.accep("տրիլիոն")
        graph_trillions = (
            (graph_hundreds + delete_space + pynutil.delete(trillions) + delete_space)
            | pynutil.insert("000", weight=0.1)
        ) + graph_billions

        graph = graph_trillions | zero

        delete_leading_zeroes = pynutil.delete(pynini.closure("0"))
        stop_at_non_zero = pynini.difference(NEMO_DIGIT, "0")
        rest_of_cardinal = pynini.closure(NEMO_DIGIT)

        clean_cardinal = delete_leading_zeroes + stop_at_non_zero + rest_of_cardinal
        clean_cardinal = clean_cardinal | "0"

        graph = graph @ clean_cardinal
        self.graph_no_exception = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
