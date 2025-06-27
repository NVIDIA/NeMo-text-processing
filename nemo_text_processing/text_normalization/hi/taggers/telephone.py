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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, delete_space, insert_space, NEMO_CHAR, NEMO_WHITE_SPACE
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

delete_zero = pynutil.delete(pynini.union("0", "०"))
delete_zero_optional = pynini.closure(delete_zero, 0, 1)
insert_shunya = pynutil.insert('शून्य') + insert_space

#Load the number mappings from the TSV file
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
std_codes = pynini.string_file(get_abs_path("data/telephone/STD_codes.tsv"))
country_codes = pynini.string_file(get_abs_path("data/telephone/country_codes.tsv"))
landline_start_digit = pynini.string_file(get_abs_path("data/telephone/landline_digits.tsv"))
mobile_start_digit = pynini.string_file(get_abs_path("data/telephone/mobile_digits.tsv"))

def load_column_from_tsv(filepath, column_index=1):
    with open(filepath, encoding='utf-8') as tsv:
        return [line.strip().split("\t")[column_index] for line in tsv if line.strip()]
    
def generate_mobile():
    country_code = (
        pynutil.insert("country_code: \"")
        + pynini.cross("+", "प्लस")
        + insert_space + country_codes
        + pynutil.insert("\" ")
        + pynini.closure(delete_space, 0, 1)
    )

    extension_optional = pynini.closure(
        pynutil.insert("extension: \"") 
        + pynini.closure(digit_to_word + insert_space, 1, 3) 
        + pynutil.insert("\" ") 
        + delete_space
        ,0,1
    )

    number_without_country = (
        pynutil.insert("number_part: \"")
        + delete_zero_optional 
        + insert_shunya 
        + mobile_start_digit + insert_space
        + pynini.closure(digit_to_word + insert_space, 9)
        + pynutil.insert("\" ") + delete_space
    )

    number_with_country = (
        country_code
        + pynutil.insert("number_part: \"")
        + mobile_start_digit + insert_space
        + pynini.closure(digit_to_word + insert_space, 9)
        + pynutil.insert("\" ") + delete_space
    ) 
    
    return (number_with_country | number_without_country) + extension_optional
    
def get_landline(std_list, std_length):
    std_digits = pynini.union(*[std for std in std_list if len(std.strip()) == std_length])
    std_graph = delete_zero_optional + insert_shunya + std_digits @ std_codes + insert_space
    
    landline_digits = pynini.closure(digit_to_word + insert_space, 1, 9-std_length) 
    landline_graph = landline_start_digit + insert_space + landline_digits
    
    seperator_optional = pynini.closure(pynini.cross("-", ""), 0, 1)

    return pynutil.insert("number_part: \"") + std_graph + seperator_optional + delete_space + landline_graph + pynutil.insert("\" ")

def generate_landline():
    std_list = load_column_from_tsv(get_abs_path("data/telephone/STD_codes.tsv"),0)
    graph = (
        get_landline(std_list, 2)
        | get_landline(std_list, 3)
        | get_landline(std_list, 4)
        | get_landline(std_list, 5)
        | get_landline(std_list, 6)
        | get_landline(std_list, 7)
    )
    
    return graph

def wrap_context(graph, keywords):
    before, after = get_context(keywords)
    return before + graph + after

def get_context(keywords: list):
    keywords = pynini.union(*keywords)

    # Define Hindi and English digits
    hindi_digits = pynini.union("०", "१", "२", "३", "४", "५", "६", "७", "८", "९")
    english_digits = pynini.union("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    all_digits = pynini.union(hindi_digits, english_digits)

    # Define word token: sequence of non-digit non-space characters followed by a space
    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(" ")

    # Limit to max 5 words
    window = pynini.closure(word, 0, 5)

    before = pynini.closure(
        pynutil.insert('context_before: "')
        + keywords
        + pynini.accep(" ")
        + window
        + pynutil.insert('" '),
        0, 1
    )

    after = pynini.closure(
        pynutil.insert('context_after: "')
        + window
        + keywords
        + pynutil.insert('" '),
        0, 1
    )

    return before.optimize(), after.optimize()

class TelephoneFst(GraphFst):
    """
    Finite state transducer for tagging telephone numbers, e.g.
        9876543210 -> telephone { number_part: "नौ आठ सात छह पाँच चार तीन दो एक शून्य" }
        +91 9876543210 -> telephone { country_code: "प्लस नौ एक", number_part: "नौ आठ सात छह पाँच चार तीन दो एक शून्य" }
        +91 9876543210 123 -> telephone { country_code: "प्लस नौ एक", number_part: "नौ आठ सात छह पाँच चार तीन दो एक शून्य", extension: "एक दो तीन" }
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        mobile_number = generate_mobile()
        mobile_number = wrap_context(mobile_number, ["नंबर", "मोबाइल", "फोन", "कॉन्टैक्ट"])

        landline = generate_landline()
        landline = wrap_context(landline, ["नंबर", "मोबाइल", "फोन", "लैंडलाइन", "कॉन्टैक्ट"])

        credit_card = (
            pynutil.insert("number_part: \"")
            + pynini.closure(digit_to_word + insert_space, 4)
            + pynutil.insert("\" ") 
            + delete_space
        )
        credit_card = wrap_context(credit_card, ["नंबर", "कार्ड", "क्रेडिट"])

        pincode = (
            pynutil.insert("number_part: \"")
            + pynini.closure(digit_to_word + insert_space, 6)
            + pynutil.insert("\" ") 
            + delete_space
        )
        pincode = wrap_context(pincode, ["नंबर", "पिन", "कोड"])

        graph = (
            pynutil.add_weight(mobile_number, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit_card, 0.9)
            | pynutil.add_weight(pincode, 1)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)

