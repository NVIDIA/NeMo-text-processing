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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_HI_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
    e.g. प्लस इक्यानवे नौ आठ सात छह पांच चार तीन दो एक शून्य => tokens { name: "+९१ ९८७६५ ४३२१०" }
    
    Args:
        Cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        hindi_digit_graph = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        hindi_digit_graph |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()

        english_digit_graph = pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()

        country_code_graph_single_digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        country_code_graph_single_digits |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        country_code_graph_single_digits |= pynini.string_file(
            get_abs_path("data/telephone/eng_to_hindi_digit.tsv")
        ).invert()

        country_code_graph_double_digits = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        country_code_graph_double_digits |= pynini.string_file(
            get_abs_path("data/telephone/teens_and_ties_eng_to_hin.tsv")
        ).invert()

        self.hindi_digit = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 9)
            + hindi_digit_graph
            + pynutil.insert("\" ")
        )
        self.english_digit = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 9)
            + english_digit_graph
            + delete_space
            + pynutil.insert("\" ")
        )

        self.country_code_with_single_digits = (
            pynutil.insert("country_code: \"")
            + pynini.closure(country_code_graph_single_digits + delete_space, 0, 2)
            + pynutil.insert("\" ")
        )
        self.country_code_with_double_digits = (
            pynutil.insert("country_code: \"")
            + pynini.closure(country_code_graph_double_digits + delete_space, 0, 1)
            + pynutil.insert("\" ")
        )
        self.country_code = self.country_code_with_single_digits | self.country_code_with_double_digits

        self.city_code_with_single_digits = (
            pynutil.insert("extension: \"")
            + pynini.closure(country_code_graph_single_digits + delete_space, 0, 2)
            + pynutil.insert("\" ")
        )
        self.city_code_with_double_digits = (
            pynutil.insert("extension: \"")
            + pynini.closure(country_code_graph_double_digits + delete_space, 0, 1)
            + pynutil.insert("\" ")
        )
        self.city_code = self.city_code_with_single_digits | self.city_code_with_double_digits

        self.landline_hindi_digit = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 6)
            + hindi_digit_graph
            + pynutil.insert("\" ")
        )
        self.landline_english_digit = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 6)
            + english_digit_graph
            + pynutil.insert("\" ")
        )

        delete_plus = pynini.union(
            pynutil.delete("प्लस") | pynutil.delete("plus") | pynutil.delete("Plus") | pynutil.delete("PLUS")
        )

        delete_zero = pynini.union(
            pynutil.delete("शून्य") | pynutil.delete("zero") | pynutil.delete("Zero") | pynutil.delete("ZERO")
        )

        graph_number_with_hindi_digit = (
            delete_plus + delete_space + self.country_code + delete_space + self.hindi_digit
        )
        graph_number_with_english_digit = delete_plus + delete_space + self.country_code + self.english_digit

        graph_landline_with_hindi_digit = (
            delete_zero + delete_space + self.city_code + delete_space + self.landline_hindi_digit
        )
        graph_landline_with_english_digit = (
            delete_zero + delete_space + self.city_code + delete_space + self.landline_english_digit
        )

        graph = (
            graph_number_with_hindi_digit
            | graph_number_with_english_digit
            | graph_landline_with_hindi_digit
            | graph_landline_with_english_digit
        )
        final_graph = self.add_tokens(graph)
        self.fst = final_graph
