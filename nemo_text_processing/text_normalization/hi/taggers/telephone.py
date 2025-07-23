# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

delete_zero = pynutil.delete(pynini.union("0", "०"))
delete_zero_optional = pynini.closure(delete_zero, 0, 1)
insert_shunya = pynutil.insert('शून्य') + insert_space

# Load the number mappings from the TSV file
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
country_codes = pynini.string_file(get_abs_path("data/telephone/country_codes.tsv"))
landline_start_digit = pynini.string_file(get_abs_path("data/telephone/landline_digits.tsv"))
mobile_start_digit = pynini.string_file(get_abs_path("data/telephone/mobile_digits.tsv"))
mobile_context = pynini.string_file(get_abs_path("data/telephone/mobile_context.tsv"))
landline_context = pynini.string_file(get_abs_path("data/telephone/landline_context.tsv"))
credit_context = pynini.string_file(get_abs_path("data/telephone/credit_context.tsv"))
pincode_context = pynini.string_file(get_abs_path("data/telephone/pincode_context.tsv"))


def generate_mobile(context_keywords):
    context_before, context_after = get_context(context_keywords)
    country_code = (
        pynutil.insert("country_code: \"")
        + context_before
        + pynini.cross("+", "प्लस")
        + insert_space
        + country_codes
        + pynutil.insert("\" ")
        + pynini.closure(delete_space, 0, 1)
    )

    extension_optional = pynini.closure(
        pynutil.insert("extension: \"")
        + pynini.closure((digit_to_word | digits | zero) + insert_space, 1, 3)
        + context_after
        + pynutil.insert("\" ")
        + delete_space,
        0,
        1,
    )

    number_without_country = (
        pynutil.insert("number_part: \"")
        + context_before
        + delete_zero_optional
        + insert_shunya
        + mobile_start_digit
        + insert_space
        + pynini.closure((digit_to_word | digits | zero) + insert_space, 9)
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )

    number_with_country = (
        country_code
        + pynutil.insert("number_part: \"")
        + mobile_start_digit
        + insert_space
        + pynini.closure((digit_to_word | digits | zero) + insert_space, 9)
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )

    return (number_with_country | number_without_country) + extension_optional


def get_landline(std_length, context_keywords):
    context_before, context_after = get_context(context_keywords)

    std_code_graph = (
        delete_zero_optional
        + insert_shunya
        + pynini.closure((digit_to_word | digits | zero) + insert_space, std_length, std_length)
    )

    landline_digit_count = 9 - std_length
    landline_graph = (
        landline_start_digit
        + insert_space
        + pynini.closure((digit_to_word | digits | zero) + insert_space, landline_digit_count, landline_digit_count)
    )

    separator_optional = pynini.closure(pynini.cross("-", ""), 0, 1)

    return (
        pynutil.insert("number_part: \"")
        + context_before
        + std_code_graph
        + separator_optional
        + delete_space
        + landline_graph
        + context_after
        + pynutil.insert("\" ")
    )


def generate_landline(context_keywords):
    graph = (
        get_landline(2, context_keywords)
        | get_landline(3, context_keywords)
        | get_landline(4, context_keywords)
        | get_landline(5, context_keywords)
        | get_landline(6, context_keywords)
        | get_landline(7, context_keywords)
    )

    return graph


def get_context(keywords: list):

    hindi_digits = pynini.union("०", "१", "२", "३", "४", "५", "६", "७", "८", "९")
    english_digits = pynini.union("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    all_digits = pynini.union(hindi_digits, english_digits)

    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(" ")

    window = pynini.closure(word, 0, 5)

    before = pynini.closure(keywords + pynini.accep(" ") + window, 0, 1)

    after = pynini.closure(pynutil.delete(" ") + window + keywords, 0, 1)

    return before.optimize(), after.optimize()


def generate_credit(context_keywords):
    context_before, context_after = get_context(context_keywords)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + pynini.closure((digit_to_word | digits | zero) + insert_space, 4)
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )


def generate_pincode(context_keywords):
    context_before, context_after = get_context(context_keywords)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + pynini.closure((digit_to_word | digits | zero) + insert_space, 6)
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )


class TelephoneFst(GraphFst):
    """
    Finite state transducer for tagging telephone numbers, e.g.
        ९१५७११४००७ -> telephone { number_part: "शून्य नौ एक पाँच सात एक एक चार शून्य शून्य सात" }
        +९१ ९२१०५१५६०६ -> telephone { country_code: "प्लस नौ एक", number_part: "नौ दो एक शून्य पाँच एक पाँच छह शून्य छह" }
        १३७४-३०९९८८ -> telephone { number_part: "शून्य एक तीन सात चार तीन शून्य नौ नौ आठ आठ" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        mobile_number = generate_mobile(mobile_context)
        landline = generate_landline(landline_context)
        credit_card = generate_credit(credit_context)
        pincode = generate_pincode(pincode_context)

        graph = (
            pynutil.add_weight(mobile_number, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit_card, 0.9)
            | pynutil.add_weight(pincode, 1)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
