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

        country_code = pynini.closure(
            pynutil.insert("country_code: \"")
            + pynini.cross("+", "प्लस")
            + insert_space + country_codes
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

        # Replace the digits with Hindi words and add spaces between words
        number_part1 = pynini.closure((NEMO_DIGIT @ digit_to_word) + insert_space, 1, 12) + delete_space

        number_part2 = pynini.closure((NEMO_DIGIT @ digit_to_word) + insert_space, 1, 10) + delete_space

        number_part3 = pynini.closure((NEMO_DIGIT @ digit_to_word) + insert_space, 1, 8) + delete_space

        # STD code using validated TSV
        std_code = (std_codes | pynini.closure((NEMO_DIGIT @ digit_to_word) + insert_space, 1, 4)) + insert_space

        # Landline number (6–8 digits)
        landline_number = pynini.closure((NEMO_DIGIT @ digit_to_word) + insert_space, 1, 8)

        # Optional hyphen or space between STD and landline
        separator = pynini.closure(pynini.accep("-") @ pynini.cross("-", " ") | delete_space, 0, 1)

        delete_punctuation = pynutil.delete(".") | pynutil.delete("।")

        # Combined STD & Landline as one number_part
        std_landline = pynutil.insert("number_part: \"") + std_code + separator + landline_number + delete_punctuation + pynutil.insert("\" ") + delete_space

        number_part = pynutil.insert("number_part: \"") + pynini.closure(number_part1 | number_part1 + number_part2 | number_part1 + number_part2 + number_part3) + pynutil.insert("\" ")

        extension = pynutil.insert("extension: \"") + pynini.closure((NEMO_DIGIT @ digit_to_word) + insert_space, 1, 3) + pynutil.insert("\" ") + delete_space

        graph = (
            std_landline |
            country_code + number_part + extension |
            country_code + number_part |
            number_part + extension |
            number_part
        )

        graph = graph.optimize()
        self.fst = self.add_tokens(graph)
