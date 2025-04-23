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
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, apply_fst


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

        std_graph = pynini.string_file(get_abs_path("data/telephone/STD_codes_eng.tsv")).invert()
        std_graph = pynini.string_file(get_abs_path("data/telephone/STD_codes_hin.tsv")).invert()
        
        landline_operator_graph = pynini.string_file(get_abs_path("data/telephone/landline_operator_digits_eng.tsv")).invert()
        landline_operator_graph |= pynini.string_file(get_abs_path("data/telephone/landline_operator_digits_hin.tsv")).invert()

        # two, three, four-digit extension code with zero
        self.city_code = (
            pynutil.insert("extension: \"")
            + std_graph 
            + delete_space
            + pynutil.insert("\" ")
        )
        
        self.city_extension = pynini.closure(self.city_code, 3, 7)
        
        # landline graph in hindi and english digits
        self.landline_hindi = (
            pynutil.insert("number_part: \"")
            + delete_space
            + landline_operator_graph
            + delete_space
            + hindi_digit_graph
            + delete_space
            + pynutil.insert("\" ")
        )
        self.landline_english = (
            pynutil.insert("number_part: \"")
            + delete_space
            + landline_operator_graph
            + delete_space
            + english_digit_graph
            + delete_space
            + pynutil.insert("\" ")
        )

        self.landline = self.landline_hindi | self.landline_english

        delete_zero = pynini.union(
            pynutil.delete("शून्य") | pynutil.delete("zero") | pynutil.delete("Zero") | pynutil.delete("ZERO")
        )
 
        graph_landline_with_extension = pynini.closure(self.city_extension + delete_space + self.landline, 11)

        graph = graph_landline_with_extension

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
        
from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
cardinal = CardinalFst()
telephone = TelephoneFst(cardinal)
#input_text = "zero one six three four two eight one eight three one" #Abohar city code(first five digits) + landline in english
input_text = "शून्य एक छह तीन चार दो आठ एक आठ तीन एक" #Abohar city code(first five digits) + landline in hindi
output = apply_fst(input_text, telephone.fst)
print(output)
