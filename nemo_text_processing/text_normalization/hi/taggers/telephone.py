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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, NEMO_DIGIT, delete_space, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

def load_column_from_tsv(filepath, column_index=1):
    with open(filepath, encoding='utf-8') as tsv:
        return [line.strip().split("\t")[column_index] for line in tsv if line.strip()]
    
#Load the number mappings from the TSV file
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
std_codes = pynini.string_file(get_abs_path("data/telephone/STD_codes.tsv"))
country_codes = pynini.string_file(get_abs_path("data/telephone/country_codes.tsv"))
landline_start_digit = pynini.string_file(get_abs_path("data/telephone/landline_digits.tsv"))
mobile_start_digit = pynini.string_file(get_abs_path("data/telephone/mobile_digits.tsv"))

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

        country_code_optional = pynini.closure(
            pynutil.insert("country_code: \"")
            + pynini.cross("+", "प्लस")
            + country_codes
            + pynutil.insert("\" ") + delete_space
            ,0,1
        )


        number_part = (
            pynutil.insert("number_part: \"")
            + mobile_start_digit + insert_space
            + pynini.closure(digit_to_word + insert_space, 9)
            + pynutil.insert("\" ") 
            + delete_space
            )

        extension_optional = pynini.closure(
            pynutil.insert("extension: \"") 
            + pynini.closure(digit_to_word + insert_space, 1, 3) 
            + pynutil.insert("\" ") 
            + delete_space
            ,0,1
            )

        mobile_number = country_code_optional + number_part + extension_optional

        def generate_landline(std_list, std_length):
            delete_zero = pynini.string_map([("0",""),("०","")])
            insert_shunya = pynutil.insert('शून्य') + insert_space
            
            std_digits = pynini.union(*[std for std in std_list if len(std.strip()) == std_length])
            std_graph = insert_shunya + std_digits @ std_codes + insert_space
            
            landline_digits = pynini.closure(digit_to_word + insert_space, 1, 9-std_length) 
            landline_graph = landline_start_digit + insert_space + landline_digits
            
            seperator_optional = pynini.closure(pynini.cross("-", " "), 0, 1)

            return pynutil.insert("number_part: \"") + std_graph + seperator_optional + delete_space + landline_graph + pynutil.insert("\" ")

        std_list = load_column_from_tsv(get_abs_path("data/telephone/STD_codes.tsv"),0)

        landline_graph = (
            generate_landline(std_list, 2)
            | generate_landline(std_list, 3)
            | generate_landline(std_list, 4)
            | generate_landline(std_list, 5)
            | generate_landline(std_list, 6)
            | generate_landline(std_list, 7)
            )

        graph = number_part | number_part + extension_optional | mobile_number | landline_graph
        
        graph = graph.optimize()
        self.fst = self.add_tokens(graph)
