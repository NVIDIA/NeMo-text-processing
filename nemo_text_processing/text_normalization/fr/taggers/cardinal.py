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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst, insert_space
from nemo_text_processing.text_normalization.fr.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "mille" }
        "2,000,000" -> cardinal { integer: "deux millions" }
    This grammar covers from single digits to hundreds of billions ("milliards" in French).
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Single digits
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))  # 1 to 9
        digits_no_one = (NEMO_DIGIT - "1") @ digits
        one = "1" @ digits

        # Double digits
        ten = pynini.string_file(get_abs_path("data/numbers/ten.tsv"))
        teens = pynini.string_file(get_abs_path("data/numbers/teens.tsv"))  # 11 to 19
        teens_no_one = (pynini.accep("1") + (NEMO_DIGIT - "1")) @ teens

        # Simple tens
        tens = pynini.string_file(get_abs_path("data/numbers/tens_simple.tsv"))
        ties_simple_unit = pynutil.insert("-") + digits_no_one
        ties_simple_one = insert_space + pynini.cross("1", "et un")
        ties_simple = tens + (pynutil.delete("0") | (ties_simple_unit | ties_simple_one))  # 20 to 69

        # Complex tens
        seventy = pynini.string_file(get_abs_path("data/numbers/seventy.tsv"))
        seventy_unit = pynutil.insert("-") + ((pynutil.insert("1") + NEMO_DIGIT) @ teens_no_one)
        seventy_one = insert_space + pynini.cross("1", "et onze")
        seventies = seventy + (pynini.cross("0", "-dix") | (seventy_unit | seventy_one))  # 70 to 79

        eighty = pynini.string_file(get_abs_path("data/numbers/eighty.tsv"))
        eighties = eighty + (pynini.cross("0", "s") | (pynutil.insert("-") + digits))  # 80 to 89

        ninety = pynini.string_file(get_abs_path("data/numbers/ninety.tsv"))
        nineties_unit = pynutil.insert("-") + ((pynutil.insert("1") + NEMO_DIGIT) @ teens)
        nineties = ninety + (pynini.cross("0", "-dix") | nineties_unit)  # 90 to 99

        all_double_digits = ten | teens | ties_simple | seventies | eighties | nineties
        self.all_double_digits = all_double_digits

        one_to_all_tens = digits | all_double_digits
        self.one_to_all_tens = one_to_all_tens.optimize()

        # Hundreds
        hundreds_parts = (pynutil.delete("0") + insert_space + digits) | (insert_space + all_double_digits)
        one_hundreds = pynini.cross("1", "cent") + (pynutil.delete("00") | hundreds_parts)
        multiple_hundreds = (digits_no_one + insert_space + pynutil.insert("cent")) + (
            pynini.cross("00", "s") | hundreds_parts
        )
        all_hundreds = one_hundreds | multiple_hundreds
        self.all_hundreds = all_hundreds

        # Thousands
        delete_separator = pynini.closure(pynutil.delete(" "), 0, 1)
        one_thousand = pynini.cross("1", "mille") + delete_separator
        other_thousands = (
            (digits_no_one | all_double_digits | all_hundreds)
            + insert_space
            + pynutil.insert("mille")
            + delete_separator
        )
        all_thousands = (
            ((one_thousand | other_thousands) + pynutil.delete("000"))
            | (one_thousand + pynutil.delete("00") + insert_space + (pynini.cross("1", "et un") | digits_no_one))
            | (other_thousands + pynutil.delete("00") + insert_space + digits)
            | ((one_thousand | other_thousands) + pynutil.delete("0") + insert_space + all_double_digits)
            | ((one_thousand | other_thousands) + insert_space + all_hundreds)
        )

        # Millions
        digits_to_hundreds_no_one = digits_no_one | all_double_digits | all_hundreds
        digits_to_thousands_no_one = digits_no_one | all_double_digits | all_hundreds | all_thousands
        millions_components = pynini.closure(delete_separator + pynini.closure(NEMO_DIGIT, 3), 2)
        delete_zeros = pynini.closure(pynutil.delete("0"), 0, 6)
        all_millions = (
            one + insert_space + pynutil.insert("million")
            | (digits_to_hundreds_no_one + insert_space + pynutil.insert("millions"))
        ) + (
            millions_components
            @ (
                delete_zeros
                + pynini.closure(insert_space + (digits_to_thousands_no_one | pynini.cross("1", "et un")), 0, 1)
            )
        )

        # Billions
        digits_to_millions_no_one = digits_to_thousands_no_one | all_millions
        billions_components = pynini.closure(delete_separator + pynini.closure(NEMO_DIGIT, 3), 3)
        delete_zeros = pynini.closure(pynutil.delete("0"), 0, 9)
        all_billions = (
            one + insert_space + pynutil.insert("milliard")
            | (digits_to_hundreds_no_one + insert_space + pynutil.insert("milliards"))
        ) + (
            billions_components
            @ (
                delete_zeros
                + pynini.closure(insert_space + (digits_to_millions_no_one | pynini.cross("1", "et un")), 0, 1)
            )
        )

        # All Numbers Union
        final_graph_masc = zero | one_to_all_tens | all_hundreds | all_thousands | all_millions | all_billions

        # Adding adjustment for fem gender (choice of gender will be random)
        final_graph_fem = final_graph_masc | (
            final_graph_masc @ (pynini.closure(NEMO_SIGMA) + pynini.cross("un", "une"))
        )
        final_graph = final_graph_fem | final_graph_masc

        self.all_nums_no_tokens = final_graph

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
