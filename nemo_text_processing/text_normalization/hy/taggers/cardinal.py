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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.hy.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        55 -> cardinal { integer: "հիսունհինգ" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        zero = pynini.string_map([("0", "զրո")])
        digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        digits_no_one = (NEMO_DIGIT - "1") @ digits

        ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).invert()
        ties_unit = digits
        double_digits = (pynini.cross("1", "տասը") | ties) + pynutil.delete("0") | (
            pynini.cross("1", "տասն") | ties
        ) + ties_unit

        self.all_double_digits = double_digits.optimize()

        one_to_all_tens = digits | double_digits
        self.one_to_all_tens = one_to_all_tens.optimize()

        hundreds_parts = (pynutil.delete("0") + insert_space + digits) | (insert_space + double_digits)
        one_hundreds = pynini.cross("1", "հարյուր") + (pynutil.delete("00") | hundreds_parts)
        multiple_hundreds = (digits_no_one + insert_space + pynutil.insert("հարյուր")) + (
            pynutil.delete("00") | hundreds_parts
        )
        all_hundreds = one_hundreds | multiple_hundreds
        self.all_hundreds = all_hundreds.optimize()

        delete_separator = pynini.closure(delete_space, 0, 1)
        one_thousand = pynini.cross("1", "հազար") + delete_separator
        other_thousands = (
            (digits_no_one | double_digits | all_hundreds) + insert_space + pynutil.insert("հազար") + delete_separator
        )
        all_thousands = (
            ((one_thousand | other_thousands) + pynutil.delete("000"))
            | (one_thousand + pynutil.delete("00") + insert_space + digits)
            | (other_thousands + pynutil.delete("00") + insert_space + digits)
            | ((one_thousand | other_thousands) + pynutil.delete("0") + insert_space + double_digits)
            | ((one_thousand | other_thousands) + insert_space + all_hundreds)
        )

        digits_to_hundreds = digits | double_digits | all_hundreds
        digits_to_thousands = digits | double_digits | all_hundreds | all_thousands
        millions_components = pynini.closure(delete_separator + pynini.closure(NEMO_DIGIT, 3), 2)
        delete_zeros = pynini.closure(pynutil.delete("0"), 0, 6)
        all_millions = (digits_to_hundreds + insert_space + pynutil.insert("միլիոն")) + (
            millions_components @ (delete_zeros + pynini.closure(insert_space + digits_to_thousands, 0, 1))
        )

        digits_to_millions = digits_to_thousands | all_millions
        billions_components = pynini.closure(delete_separator + pynini.closure(NEMO_DIGIT, 3), 3)
        delete_zeros = pynini.closure(pynutil.delete("0"), 0, 9)
        all_billions = (digits_to_hundreds + insert_space + pynutil.insert("միլիարդ")) + (
            billions_components @ (delete_zeros + pynini.closure(insert_space + digits_to_millions, 0, 1))
        )

        final_graph = zero | digits | double_digits | all_hundreds | all_thousands | all_millions | all_billions
        self.all_nums_no_tokens = final_graph

        final_graph = pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        self.final_graph = final_graph
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
