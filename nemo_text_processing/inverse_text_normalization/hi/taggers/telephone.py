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
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path

shunya = pynini.cross("शून्य", "०")

digit = (
    pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert() 
    | pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()
    )

def get_context(keywords: list):
    keywords = pynini.union(*keywords)

    # TODO: create a tsv for below data
    hindi_digits = pynini.union("शून्य", "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छे", 'छह', "सात", "आठ", "नौ")
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
    digits = pynini.closure(digit + delete_space, length-1, length-1) + digit
    
    graph_after_context = digits + NEMO_WHITE_SPACE + context_after
    graph_before_context = context_before + NEMO_WHITE_SPACE + digits
    graph_without_context = digits
        
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

    country_code = pynini.cross("प्लस", "+") + pynini.closure(delete_space + digit, 2, 2) + NEMO_WHITE_SPACE
    graph_country_code = (
        pynutil.insert("country_code: \"")
        + (context_before + NEMO_WHITE_SPACE)**(0, 1)
        + country_code
        + pynutil.insert("\" ")
    )
    number_part = pynini.closure(digit + delete_space, 9, 9) + digit
    graph_number = (
        pynutil.insert("number_part: \"")
        + number_part
        + pynini.closure(NEMO_WHITE_SPACE + context_after, 0, 1)
        + pynutil.insert("\" ")
    )

    graph = graph_country_code + graph_number 
    return graph.optimize()
    
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
    from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
    def run_test(tests, graph):
        test_count = len(tests)
        fail_count = pass_count = 0
        print()
        for test in tests:
            try:
                result = pynini.shortestpath(test @ graph).string()
                print(result)
                print('-'*50)
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
        # "एक एक एक एक एक एक एक", # should fail
        # "एक एक एक एक एक", # should fail
    ]

    credit_tests = [
        "क्रेडिट एक एक एक एक", 
        "एक एक एक एक क्रेडिट",
        "एक एक एक एक",
        # "एक एक एक एक एक", # should fail
        # "एक एक एक", # should fail 
    ]

    mobile_tests = [
        "आठ चार तीन सात दो शून्य पांच छह एक आठ",
        "मोबाइल प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ",
        "मोबाइल प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ छह एक आठ", # extention 
        "आठ चार तीन सात दो शून्य पांच छह एक आठ छह एक आठ", # extention 
        "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ मोबाइल",
        "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ",
        # "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक आठ आठ आठ आठ आठ", # should fail
        # "प्लस नौ एक आठ चार तीन सात दो शून्य पांच छह एक", # should fail
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

    # run_test(["प्लस नौ एक"], mobile)
    # run_test(["एक एक एक एक एक एक एक एक एक एक"], mobile)
    run_test(["प्लस नौ एक एक एक एक एक एक एक एक एक एक एक"], TelephoneFst(CardinalFst()).fst)

    # python nemo_text_processing\inverse_text_normalization\inverse_normalize.py --text="प्लस नौ एक नौ आठ सात छह पाँच चार तीन दो एक शून्य" --verbose --overwrite_cache 
    # pytest tests\nemo_text_processing\hi\test_telephone.py --cpu --cache-clear