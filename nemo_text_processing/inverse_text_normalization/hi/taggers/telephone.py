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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import GraphFst, delete_space, NEMO_WHITE_SPACE, NEMO_CHAR
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, apply_fst

shunya = pynini.cross("शून्य", "०")

digit = (
    pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert() 
    | pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()
    )

country_code = pynini.cross("नौ एक", "९१")
mobile_start_digit = pynini.string_file(get_abs_path("data/telephone/mobile_digits.tsv")).invert()

def get_context(keywords: list):
    keywords = pynini.union(*keywords)

    hindi_digits = pynini.union("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छे", "सात", "आठ", "नौ")
    english_digits = pynini.union("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    all_digits = hindi_digits | english_digits

    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + NEMO_WHITE_SPACE
    window = pynini.closure(word, 0, 5)
    before = (keywords + window).optimize()
    after = (window + keywords).optimize()

    return before, after

def generate_context_graph(context_keywords, length):
    context_before, context_after = get_context(context_keywords)
    graph_after_context = pynini.closure(digit + NEMO_WHITE_SPACE, length) + context_after
    graph_before_context = context_before + NEMO_WHITE_SPACE + pynini.closure(digit + NEMO_WHITE_SPACE, length-1, length-1) + digit
    graph_without_context = pynini.closure(digit + NEMO_WHITE_SPACE, length-1, length-1) + digit
        
    return (
        pynutil.insert("number_part: \"")
        + (
            graph_before_context 
            | graph_after_context
            | graph_without_context 
        )
        + pynutil.insert("\" ")
    ).optimize()

def generate_pincode(context_keywords):
    return generate_context_graph(context_keywords, 6)

def generate_credit(context_keywords):
    return generate_context_graph(context_keywords, 4)

def generate_mobile(context_keywords):
    context_before, context_after = get_context(context_keywords)

    graph_country_code = (
        pynutil.insert("country_code: \"")
        + pynini.closure(context_before + NEMO_WHITE_SPACE, 0, 1) 
        + pynini.cross("प्लस", "+") + NEMO_WHITE_SPACE
        + delete_space + country_code
        + pynutil.insert("\" ")
    )

    number_without_country = (
        pynutil.insert("number_part: \"")
        + pynini.closure(context_before + NEMO_WHITE_SPACE, 0, 1)
        + mobile_start_digit + delete_space
        + pynini.closure(digit + delete_space, 8, 8) + digit
        + pynini.closure(context_after, 0, 1)
        + pynutil.insert("\" ")
    )

    number_with_country = (
        graph_country_code + NEMO_WHITE_SPACE
        + pynutil.insert("number_part: \"")
        + mobile_start_digit + delete_space
        + pynini.closure(digit + delete_space, 8, 8) + digit
        + pynini.closure(NEMO_WHITE_SPACE + context_after, 0, 1)
        + pynutil.insert("\" ")
    )

    ext_digits = pynini.closure(digit + delete_space, 0, 2) + digit
    extension = (
        pynutil.insert("extension: \"")
        + delete_space                                     
        + pynini.closure(ext_digits, 0, 1)
        + pynini.closure(context_after, 0, 1)                 
        + pynutil.insert("\" ")
        + delete_space
    )

    return (number_without_country | number_with_country) + extension
    
def generate_telephone(context_keywords):
    context_before, context_after = get_context(context_keywords)
    shunya_optional = pynini.closure(shunya + delete_space, 0, 1)

    landline = shunya_optional + pynini.closure(digit + delete_space, 9, 9) + digit
    landline_with_context_before = context_before + NEMO_WHITE_SPACE + landline
    landline_with_context_after = landline + NEMO_WHITE_SPACE + context_after

    return (
        pynutil.insert("number_part: \"") 
        + (landline | landline_with_context_before | landline_with_context_after) 
        + pynutil.insert("\" ")
    )

class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
    e.g. प्लस इक्यानवे नौ आठ सात छह पांच चार तीन दो एक शून्य => tokens { name: "+९१ ९८७६५ ४३२१०" }

    Args:
        Cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        mobile = generate_mobile(["नंबर", "मोबाइल", "फोन", "कॉल"])
        landline = generate_telephone(["नंबर", "मोबाइल", "फोन", "लैंडलाइन", "कॉल"])
        pincode = generate_pincode(["पिन", "कोड", "पिनकोड"])
        credit = generate_credit(["नंबर", "कार्ड", "क्रेडिट"])

        graph = (
            pynutil.add_weight(mobile, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit, 0.9)
            | pynutil.add_weight(pincode, 1)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)

if __name__ == '__main__':
    def run_test(tests, graph):
        test_count = len(tests)
        fail_count = pass_count = 0
        print()
        for test in tests:
            try:
                # print(pynini.shortestpath(test @ graph).string())
                # print('-'*50)
                pass_count += 1
            except pynini.FstOpError:
                print(f"Error: No valid output with given input: '{test}'")
                fail_count += 1
                print('-'*50)
        
        print(f"\nTotal : {test_count} \nFailed : {fail_count} \nPassed : {pass_count}")

    mobile = generate_mobile(["नंबर", "मोबाइल", "फोन", "कॉल"])
    landline = generate_telephone(["नंबर", "मोबाइल", "फोन", "लैंडलाइन", "कॉल"])
    pincode = generate_pincode(["पिन", "कोड", "पिनकोड"])
    credit = generate_credit(["नंबर", "कार्ड", "क्रेडिट"])

    pincode_tests = [
        "पिनकोड एक एक एक एक एक एक", 
        "एक एक एक एक एक एक पिनकोड",
        "एक एक एक एक एक एक",
        "एक एक एक एक एक एक एक", # should fail
        "एक एक एक एक एक", # should fail
    ]

    credit_tests = [
        "क्रेडिट एक एक एक एक", 
        "एक एक एक एक क्रेडिट",
        "एक एक एक एक",
        "एक एक एक एक एक", # should fail
        "एक एक एक", # should fail 
    ]

    mobile_tests = [
        "आठ चार तीन सात दो शून्य पांच छह एक आठ",
        "मोबाइल प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ",
        "मोबाइल प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ छह एक आठ", # extention 
        "आठ चार तीन सात दो शून्य पांच छह एक आठ छह एक आठ", # extention 
        "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ मोबाइल",
        "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ",
        "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ आठ आठ आठ आठ", # should fail
        "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक", # should fail
    ]

    telephone_tests = [
        "शून्य चार शून्य दो सात आठ एक आठ तीन नौ नौ",
        "चार शून्य दो सात आठ एक आठ तीन नौ नौ",

        "मोबाइल शून्य चार शून्य दो सात आठ एक आठ तीन नौ नौ",
        "शून्य चार शून्य दो सात आठ एक आठ तीन नौ नौ मोबाइल",
        "मोबाइल चार शून्य दो सात आठ एक आठ तीन नौ नौ",
        "चार शून्य दो सात आठ एक आठ तीन नौ नौ मोबाइल",
    ]

    tests = mobile_tests + pincode_tests + credit_tests + telephone_tests
    combined_graph = mobile | pincode | credit | landline

    run_test(telephone_tests, landline)