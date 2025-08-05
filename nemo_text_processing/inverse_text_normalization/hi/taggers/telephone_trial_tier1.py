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
    Finite state transducer for classifying telephone numbers.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")
        
        english_digit_graph = pynini.string_file(get_abs_path("data/telephone/eng_to_hindi_digit.tsv")).invert()
        hindi_digit_graph = cardinal.graph_digit | cardinal.graph_zero
        digit = english_digit_graph | hindi_digit_graph
        
        # Loading data files
        std_tier1_hin = load_column_from_tsv(get_abs_path("data/telephone/std_tier1_hin.tsv"))
        std_tier1_eng = load_column_from_tsv(get_abs_path("data/telephone/std_tier1_eng.tsv"))
        
        # Construct the tier1 graphs for Hindi and English
        std_tier1_hin_graph = pynini.union(*[pynini.string_map([(x.split()[0], x.split()[1])]) for x in std_tier1_hin])
        std_tier1_eng_graph = pynini.union(*[pynini.string_map([(x.split()[0], x.split()[1])]) for x in std_tier1_eng])

        # Combine them into the tier1 code graph
        tier1_code_graph = std_tier1_hin_graph | std_tier1_eng_graph
        allowed_start_hindi_landline = pynini.union("दो", "तीन", "चार", "छः", "छह", "छे")
        landline_start_hindi_graph = allowed_start_hindi_landline @ hindi_digit_graph

        allowed_start_english_landline = pynini.union("two", "three", "four", "six")
        landline_start_english_graph = allowed_start_english_landline @ english_digit_graph

        # Combine the Hindi and English landline start digits
        landline_start_digit = landline_start_hindi_graph + landline_start_english_graph

        # Combine the start digits with the digit graph
        graph_landline_start_digit = (landline_start_digit @ digit) + delete_space

        # Filter valid two-part entries and create corresponding FSTs for tier1_std
        tier1_std = pynini.union(*[
            pynini.string_map([(entry.split()[0], entry.split()[1])])
            for entry in filter(lambda x: len(x.split()) == 2, std_tier1_hin + std_tier1_eng)
        ])

        # Landline start digit graph (Hindi and English)
        graph_valid_landline_start_digit = landline_start_digit  # landline_start_digit is already an FST
        graph_landline_start_digit = (graph_valid_landline_start_digit @ digit) + delete_space

        # Final two_digit_graph construction for tier1 landline processing
        two_digit_graph = (
            (pynutil.insert("extension: \"") + (tier1_std @ tier1_code_graph) + pynutil.insert("\" "))
            + delete_space
            + (pynutil.insert("number_part: \"") + graph_landline_start_digit + pynini.closure((digit + delete_space), 7, 7) + pynutil.insert("\" "))
        ).optimize()
        delete_zero = pynini.union(
            pynutil.delete("शून्य") | pynutil.delete("zero") | pynutil.delete("Zero") | pynutil.delete("ZERO")
        )
        

        # Combine everything into the final graph for tier1 landline
        graph_landline_tier1 = delete_zero + delete_space + two_digit_graph        
        graph = graph_landline_tier1 #| graph_landline | graph_mobile
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
