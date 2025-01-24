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
from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst, get_abs_path


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

        # two, three, four-digit extension code with zero
        self.city_two_digit_code_hindi = (
            pynutil.insert("extension: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 2)
            + pynutil.insert("\" ")
        )
        self.city_two_digit_code_english = (
            pynutil.insert("extension: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 2)
            + pynutil.insert("\" ")
        )
        self.city_three_digit_code_hindi = (
            pynutil.insert("extension: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 3)
            + pynutil.insert("\" ")
        )
        self.city_three_digit_code_english = (
            pynutil.insert("extension: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 3)
            + pynutil.insert("\" ")
        )
        self.city_four_digit_code_hindi = (
            pynutil.insert("extension: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 4)
            + pynutil.insert("\" ")
        )
        self.city_four_digit_code_english = (
            pynutil.insert("extension: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 4)
            + pynutil.insert("\" ")
        )

        # concise extensions graphs - 2,3,4-digit
        self.city_two_digit_extension = self.city_two_digit_code_hindi | self.city_two_digit_code_english
        self.city_three_digit_extension = self.city_three_digit_code_hindi | self.city_three_digit_code_english
        self.city_four_digit_extension = self.city_four_digit_code_hindi | self.city_four_digit_code_english

        # 7-digit landline graph for 2-digit extension in hindi and english digits
        self.landline_with_extension_two_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 6)
            + pynutil.insert("\" ")
        )
        self.landline_with_extension_two_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 6)
            + pynutil.insert("\" ")
        )
        self.landline_two = self.landline_with_extension_two_hindi | self.landline_with_extension_two_english

        # 7-digit landline graph for 3-digit extension in hindi and english digits
        self.landline_with_extension_three_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 7)
            + pynutil.insert("\" ")
        )
        self.landline_with_extension_three_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 7)
            + pynutil.insert("\" ")
        )
        self.landline_three = self.landline_with_extension_three_hindi | self.landline_with_extension_three_english

        # 7-digit landline graph for 4-digit extension in hindi and english digits
        self.landline_with_extension_four_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 8)
            + pynutil.insert("\" ")
        )
        self.landline_with_extension_four_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 8)
            + pynutil.insert("\" ")
        )
        self.landline_four = self.landline_with_extension_four_hindi | self.landline_with_extension_four_english

        self.pincode_in_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 5)
            + hindi_digit_graph
            + pynutil.insert("\" ")
        )
        self.pincode_in_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 5)
            + english_digit_graph
            + pynutil.insert("\" ")
        )

        self.credit_card_last_four_digits_in_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hindi_digit_graph + delete_space, 0, 3)
            + hindi_digit_graph
            + pynutil.insert("\" ")
        )
        self.credit_card_last_four_digits_in_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(english_digit_graph + delete_space, 0, 3)
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

        graph_landline_with_two_digit_extension = (
            delete_zero + delete_space + self.city_two_digit_extension + delete_space + self.landline_two
        )
        graph_landline_with_three_digit_extension = (
            delete_zero + delete_space + self.city_three_digit_extension + delete_space + self.landline_three
        )
        graph_landline_with_four_digit_extension = (
            delete_zero + delete_space + self.city_four_digit_extension + delete_space + self.landline_four
        )

        graph_pincode = self.pincode_in_hindi | self.pincode_in_english

        graph_credit_card_last_four_digits = (
            self.credit_card_last_four_digits_in_hindi | self.credit_card_last_four_digits_in_english
        )

        graph = (
            graph_number_with_hindi_digit
            | graph_number_with_english_digit
            | graph_landline_with_two_digit_extension
            | graph_landline_with_three_digit_extension
            | graph_landline_with_three_digit_extension
            | graph_pincode
            | graph_credit_card_last_four_digits
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph


# from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
# cardinal = CardinalFst()
# telephone = TelephoneFst(cardinal)

# input_text = "zero eight zero two two nine four one one one" # zero+ two digit extension + landline in english
# input_text = "zero eight zero nine two two nine four one one one" # zero + three digit extension + landline in english
# input_text = "zero eight zero nine one two two nine four one one one" #zero + four digit extension + landline in english

# input_text = "शून्य सात नौ एक नौ आठ सात छह पांच चार" #zero + two digit extension + landline in hindi
# input_text = "शून्य सात नौ नौ एक नौ आठ सात छह पांच चार" #zero + three digit extension + landline in hindi
# input_text = "शून्य सात नौ एक एक एक नौ आठ सात छह पांच चार" #zero+ four digit digit extension + landline in hindi

# output = apply_fst(input_text, telephone.fst)
# print(output)
