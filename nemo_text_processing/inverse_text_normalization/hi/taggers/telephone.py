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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, load_column_from_tsv


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
    e.g. प्लस इक्यानवे नौ आठ सात छह पांच चार तीन दो एक शून्य => tokens { name: "+९१ ९८७६५ ४३२१०" }

    Args:
        Cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")
        eng_to_hin_digit_graph = pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()
        hin_word_to_digit_graph = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        hin_word_to_digit_graph |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        digit = eng_to_hin_digit_graph | hin_word_to_digit_graph

        eng_word_to_hin_std_graph = pynini.string_file(get_abs_path("data/telephone/STD_codes_eng.tsv")).invert()
        hin_word_to_hin_std_graph = pynini.string_file(get_abs_path("data/telephone/STD_codes_hin.tsv")).invert()
        words_to_hin_std_graph = eng_word_to_hin_std_graph | hin_word_to_hin_std_graph

        graph_eng_stds = load_column_from_tsv(get_abs_path("data/telephone/STD_codes_eng.tsv"))
        graph_hin_stds = load_column_from_tsv(get_abs_path("data/telephone/STD_codes_hin.tsv"))
        graph_valid_stds = graph_eng_stds + graph_hin_stds

        graph_eng_valid_landline_start_digits = load_column_from_tsv(
            get_abs_path("data/telephone/landline_operator_digits_eng.tsv")
        )
        graph_hin_valid_landline_start_digits = load_column_from_tsv(
            get_abs_path("data/telephone/landline_operator_digits_hin.tsv")
        )
        graph_valid_landline_start_digits = (
            graph_eng_valid_landline_start_digits + graph_hin_valid_landline_start_digits
        )

        landline_start_digits = pynini.union(*graph_valid_landline_start_digits)
        landline_start_digit = (landline_start_digits @ digit) + delete_space

        two_digit_std = pynini.union(*list(filter(lambda x: len(x.split()) == 2, graph_valid_stds)))
        two_digit_graph = (
            (pynutil.insert("extension: \"") + (two_digit_std @ words_to_hin_std_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + landline_start_digit
                + pynini.closure((digit + delete_space), 7, 7)
                + pynutil.insert("\" ")
            )
        ).optimize()

        three_digit_std = pynini.union(*list(filter(lambda x: len(x.split()) == 3, graph_valid_stds)))
        three_digit_std_graph = (
            (pynutil.insert("extension: \"") + (three_digit_std @ words_to_hin_std_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + landline_start_digit
                + pynini.closure((digit + delete_space), 6, 6)
                + pynutil.insert("\" ")
            )
        ).optimize()

        four_digit_std = pynini.union(*list(filter(lambda x: len(x.split()) == 4, graph_valid_stds)))
        four_digit_std_graph = (
            (pynutil.insert("extension: \"") + (four_digit_std @ words_to_hin_std_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + landline_start_digit
                + pynini.closure((digit + delete_space), 5, 5)
                + pynutil.insert("\" ")
            )
        ).optimize()

        five_digit_std = pynini.union(*list(filter(lambda x: len(x.split()) == 5, graph_valid_stds)))
        five_digit_std_graph = (
            (pynutil.insert("extension: \"") + (five_digit_std @ words_to_hin_std_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + landline_start_digit
                + pynini.closure((digit + delete_space), 4, 4)
                + pynutil.insert("\" ")
            )
        ).optimize()

        six_digit_std = pynini.union(*list(filter(lambda x: len(x.split()) == 6, graph_valid_stds)))
        six_digit_std_graph = (
            (pynutil.insert("extension: \"") + (six_digit_std @ words_to_hin_std_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + landline_start_digit
                + pynini.closure((digit + delete_space), 3, 3)
                + pynutil.insert("\" ")
            )
        ).optimize()

        seven_digit_std = pynini.union(*list(filter(lambda x: len(x.split()) == 7, graph_valid_stds)))
        seven_digit_std_graph = (
            (pynutil.insert("extension: \"") + (seven_digit_std @ words_to_hin_std_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + landline_start_digit
                + pynini.closure((digit + delete_space), 2, 2)
                + pynutil.insert("\" ")
            )
        ).optimize()

        delete_zero = pynini.union(
            pynutil.delete("शून्य") | pynutil.delete("zero") | pynutil.delete("Zero") | pynutil.delete("ZERO")
        )

        graph_landline = pynutil.add_weight(
            delete_zero
            + delete_space
            + (
                two_digit_graph
                | three_digit_std_graph
                | four_digit_std_graph
                | five_digit_std_graph
                | six_digit_std_graph
                | seven_digit_std_graph
            ),
            -100,
        )

        # mobile numbers
        eng_word_to_hin_country_graph = pynini.string_file(
            get_abs_path("data/telephone/country_codes_eng.tsv")
        ).invert()
        hin_word_to_hin_country_graph = pynini.string_file(
            get_abs_path("data/telephone/country_codes_hin.tsv")
        ).invert()
        words_to_hin_country_graph = eng_word_to_hin_country_graph | hin_word_to_hin_country_graph

        graph_eng_country_code = load_column_from_tsv(get_abs_path("data/telephone/country_codes_eng.tsv"))
        graph_hin_country_code = load_column_from_tsv(get_abs_path("data/telephone/country_codes_hin.tsv"))
        graph_valid_country_code = graph_eng_country_code + graph_hin_country_code

        graph_eng_valid_mobile_start_digits = load_column_from_tsv(
            get_abs_path("data/telephone/mobile_operator_digits_eng.tsv")
        )
        graph_hin_valid_mobile_start_digits = load_column_from_tsv(
            get_abs_path("data/telephone/mobile_operator_digits_hin.tsv")
        )
        graph_valid_mobile_start_digits = graph_eng_valid_mobile_start_digits + graph_hin_valid_mobile_start_digits

        mobile_start_digits = pynini.union(*graph_valid_mobile_start_digits)
        mobile_start_digit = (mobile_start_digits @ digit) + delete_space

        country_code = pynini.union(*list(filter(lambda x: len(x.split()) == 2, graph_valid_country_code)))
        country_code_graph = (
            (pynutil.insert("country_code: \"") + (country_code @ words_to_hin_country_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + mobile_start_digit
                + pynini.closure((digit + delete_space), 9, 9)
                + pynutil.insert("\" ")
            )
        ).optimize()
        mobile_graph = (
            (
                pynutil.insert("number_part: \"")
                + mobile_start_digit
                + pynini.closure((digit + delete_space), 9, 9)
                + pynutil.insert("\" ")
            )
        ).optimize()

        delete_plus = pynini.union(
            pynutil.delete("प्लस") | pynutil.delete("plus") | pynutil.delete("Plus") | pynutil.delete("PLUS")
        )

        graph_mobile = pynutil.add_weight(delete_plus + delete_space + country_code_graph, -100)
        graph_mobile |= pynutil.add_weight(mobile_graph, -100)

        # Pincode
        self.pincode_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hin_word_to_digit_graph + delete_space, 0, 5)
            + hin_word_to_digit_graph
            + pynutil.insert("\" ")
        )
        self.pincode_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(eng_to_hin_digit_graph + delete_space, 0, 5)
            + eng_to_hin_digit_graph
            + pynutil.insert("\" ")
        )

        graph_pincode = self.pincode_hindi | self.pincode_english

        # last digits of credit card
        self.credit_card_last_digits_hindi = (
            pynutil.insert("number_part: \"")
            + pynini.closure(hin_word_to_digit_graph + delete_space, 0, 3)
            + hin_word_to_digit_graph
            + pynutil.insert("\" ")
        )
        self.credit_card_last_digits_english = (
            pynutil.insert("number_part: \"")
            + pynini.closure(eng_to_hin_digit_graph + delete_space, 0, 3)
            + eng_to_hin_digit_graph
            + pynutil.insert("\" ")
        )

        graph_credit_card_last_digits = self.credit_card_last_digits_hindi | self.credit_card_last_digits_english

        graph = graph_landline | graph_mobile | graph_pincode | graph_credit_card_last_digits
        final_graph = self.add_tokens(graph)
        self.fst = final_graph