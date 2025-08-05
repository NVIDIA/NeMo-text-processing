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
    DEVANAGARI_DIGIT,
    integer_to_devanagari,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, load_column_from_tsv, apply_fst


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
    e.g. प्लस नौ एक नौ आठ सात छह पांच चार तीन दो एक शून्य => tokens { name: "+९१ ९८७६५ ४३२१०" }
    
    Args:
        Cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")
        
        english_digit_graph = pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()
        hindi_digit_graph = cardinal.graph_digit | cardinal.graph_zero
        digit = english_digit_graph | hindi_digit_graph
        
        std_tier1_hin_code = pynini.string_file(get_abs_path("data/telephone/std_tier1_hin.tsv")).invert()
        std_tier1_eng_code = pynini.string_file(get_abs_path("data/telephone/std_tier1_eng.tsv")).invert()
        tier1_code_graph = std_tier1_hin_code | std_tier1_eng_code
        
        std_tier1_hin = load_column_from_tsv(get_abs_path("data/telephone/std_tier1_hin.tsv"))
        std_tier1_eng = load_column_from_tsv(get_abs_path("data/telephone/std_tier1_eng.tsv"))
        std_tier1_graph = std_tier1_hin + std_tier1_eng
        
        allowed_start_hindi_landline = pynini.union("दो", "तीन", "चार", "छः", "छह", "छे")
        landline_start_hindi_graph = allowed_start_hindi_landline @ hindi_digit_graph 
        allowed_start_english_landline = pynini.union("two", "three", "four", "six")
        landline_start_english_graph = allowed_start_english_landline @ english_digit_graph
        landline_start_digit = landline_start_hindi_graph + landline_start_english_graph
        
        graph_valid_landline_start_digit = pynini.union(*landline_start_digit)
        graph_landline_start_digit = (graph_valid_landline_start_digit @ digit) + delete_space
        
        tier1_std = pynini.union(*list(filter(lambda x: len(x.split()) == 2, std_tier1_graph)))
        two_digit_graph = (
            (pynutil.insert("extension: \"") + (tier1_std @ tier1_code_graph) + pynutil.insert("\" "))
            + delete_space
            + (
                pynutil.insert("number_part: \"")
                + graph_landline_start_digit
                + pynini.closure((digit + delete_space), 7, 7)
                + pynutil.insert("\" ")
            )
        ).optimize()
        
        self.city_code = (
            pynutil.insert("extension: \"")
            + pynini.closure(digit + delete_space, 3, 3)
            + pynutil.insert("\" ")
        )
        
        self.landline_hindi = (
            pynutil.insert("number_part: \"")
            + landline_start_hindi_graph + delete_space
            + pynini.closure(hindi_digit_graph + delete_space, 6, 6)
            + pynutil.insert("\" ")
        )
        self.landline_english = (
            pynutil.insert("number_part: \"")
            + landline_start_english_graph + delete_space
            + pynini.closure(english_digit_graph + delete_space, 6, 6)
            + pynutil.insert("\" ")
        )
        self.landline = self.landline_hindi | self.landline_english
        
        delete_zero = pynini.union(
            pynutil.delete("शून्य") | pynutil.delete("zero") | pynutil.delete("Zero") | pynutil.delete("ZERO")
        )
        
        #mobile numbers
        country_code_hindi = cardinal.graph_digit | cardinal.graph_zero
        country_code_english = pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()
        country_code = country_code_hindi | country_code_english
        
        allowed_start_hindi_mobile = pynini.union("छः", "छह", "छे", "सात", "आठ", "नौ")
        mobile_start_hindi_graph = allowed_start_hindi_mobile @ hindi_digit_graph
        allowed_start_english_mobile = pynini.union("six", "seven", "eight", "nine")
        mobile_start_english_graph = allowed_start_english_mobile @ english_digit_graph
        
        self.country_code_hindi = (
            pynutil.insert("country_code: \"")
            + pynini.closure(country_code_hindi + delete_space, 0, 2)
            + pynutil.insert("\" ")
        )
        self.country_code_english = (
            pynutil.insert("country_code: \"")
            + pynini.closure(country_code_english + delete_space, 0, 2)
            + pynutil.insert("\" ")
        )
        
        self.hindi_digit = (
            pynutil.insert("number_part: \"")
            + mobile_start_hindi_graph + delete_space
            + pynini.closure(digit + delete_space, 0, 9)
            + hindi_digit_graph
            + pynutil.insert("\" ")
        )
        self.english_digit = (
            pynutil.insert("number_part: \"")
            + mobile_start_english_graph + delete_space
            + pynini.closure(digit + delete_space, 0, 9)
            + english_digit_graph
            + pynutil.insert("\" ")
        )
        
        delete_plus = pynini.union(
            pynutil.delete("प्लस") | pynutil.delete("plus") | pynutil.delete("Plus") | pynutil.delete("PLUS")
        )
        
        graph_landline_tier1 = delete_zero + delete_space + two_digit_graph
        
        graph_landline = delete_zero + delete_space + self.city_code + delete_space + self.landline
        
        graph_mobile_hindi = delete_plus + delete_space + self.country_code_hindi + delete_space + self.hindi_digit
        graph_mobile_hindi |= self.hindi_digit
        graph_mobile_english = delete_plus + delete_space + self.country_code_english + delete_space + self.english_digit
        graph_mobile_english |= self.english_digit
        
        graph_mobile = graph_mobile_hindi | graph_mobile_english
        
        graph = graph_landline_tier1 | graph_landline | graph_mobile
        final_graph = self.add_tokens(graph)
        self.fst = final_graph
                    
from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
cardinal = CardinalFst()
telephone = TelephoneFst(cardinal)
#input_text = "zero eight nine one six nine four one one one two"
#input_text = "plus nine one nine four nine one six one one one seven seven"
#input_text = "nine four nine one six one one one seven seven"
input_text = "शून्य आठ शून्य दो नौ चार एक एक एक दो एक" #- tier 1 city (not yet implemented)
#input_text = "प्लस  नौ एक नौ आठ सात छह पांच चार तीन दो एक शून्य"
#input_text = "नौ आठ सात छह पांच चार तीन दो एक शून्य"
#input_text = "शून्य दो चार शून्य तीन सात एक चार पांच चार तीन"
output = apply_fst(input_text, telephone.fst)
print(output)
