# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
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
import string
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst,NEMO_CHAR,insert_space
from nemo_text_processing.text_normalization.rw.utils import get_abs_path

def apply_fst(text, fst):
    try:
        print(pynini.shortestpath(text @ fst).string())
        print(len(pynini.shortestpath(text @ fst).string()))

    except pynini.FstOpError:
        print(f"Error: no valid output with given'input: '{text}'")

class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        alphabet = string.ascii_letters
        rewrite_na_fst = pynini.cdrewrite(pynini.cross(" "," na "),pynini.union(*"aeiouAEIOU "),pynini.union(*"BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz"),NEMO_CHAR.closure())
        rewrite_n_fst = pynini.cdrewrite(pynini.cross(" "," n'"),pynini.union(*"aeiouAEIOU "),pynini.union(*"aeiouAEIOU"),NEMO_CHAR.closure())
        remove_underscore_fst = pynini.cdrewrite(pynini.cross("_"," "),pynini.union(*alphabet),pynini.union(*alphabet),NEMO_CHAR.closure())
        remove_extra_space_fst = pynini.cdrewrite(pynini.cross("  "," "),pynini.union(*alphabet),pynini.union(*alphabet),NEMO_CHAR.closure())
        remove_trailing_space_fst = pynini.cdrewrite(pynini.cross(pynini.accep(' ').closure(),''),pynini.union(*alphabet).closure(),'[EOS]',NEMO_CHAR.closure())

        rewrite_add_separator_fst = pynini.compose(rewrite_na_fst,rewrite_n_fst)
        ten_thousand = pynini.string_map([("ibihumbi_icumi","10")])
        ten = pynini.string_map([("icumi","10")])
        digits = pynini.string_map([
            ("rimwe","1"),
            ("kabiri","2"),
            ("gatatu","3"),
            ("kane","4"),
            ("gatanu","5"),
            ("gatandatu","6"),
            ("karindwi","7"),
            ("umunani","8"),
            ("icyenda","9"),
        ])
        digits_for_thousands = pynini.string_map([
            ("","0"),
            ("kimwe","1"),
            ("bibiri","2"),
            ("bitatu","3"),
            ("bine","4"),
            ("bitanu","5"),
            ("bitandatu","6"),
            ("birindwi","7"),
            ("umunani","8"),
            ("icyenda","9")
        ]) 
        digits_millions_trillions= pynini.string_map([
            ("","0"),
            ("imwe","1"),
            ("ebyiri","2"),
            ("eshatu","3"),
            ("enye","4"),
            ("eshanu","5"),
            ("esheshatu","6"),
            ("zirindwi","7"),
            ("umunani","8"),
            ("icyenda","9")
        ]) 
        tens = pynini.string_map([
            (" ","0"),
            ("makumyabiri","2"),
            ("mirongo_itatu","3"),
            ("mirongo_ine","4"),
            ("mirongo_itanu","5"),
            ("mirongo_itandatu","6"),
            ("mirongo_irindwi","7"),
            ("mirongo_inani","8"),
            ("mirongo_icyenda","9")
        ])
        tens_for_ends = pynini.string_map([("icumi","1")])|tens 
        tens_for_beginnings= pynini.string_map([("cumi","1")])|tens
        hundreds = pynini.string_map([
            ("ijana","1"),
            ("magana_abiri","2"),
            ("magana_atatu","3"),
            ("magana_ane","4"),
            ("magana_atanu","5"),
            ("magana_atandatu","6"),
            ("magana_arindwi","7"),
            ("magana_inani","8"),
            ("magana_cyenda","9")
        ])
        thousands = pynini.string_map([
            ("igihumbi","1"),
            ("ibihumbi_bibiri","2"),
            ("ibihumbi_bitatu","3"),
            ("ibihumbi_bine","4"),
            ("ibihumbi_bitanu","5"),
            ("ibihumbi_bitandatu","6"),
            ("ibihumbi_birindwi","7"),
            ("ibihumbi_umunani","8"),
            ("ibihumbi_icyenda","9")
        ])
        tens_of_thousands = pynini.string_map([
            ("ibihumbi_cumi","1"),
            ("ibihumbi_makumyabiri","2"),
            ("ibihumbi_mirongo_itatu","3"),
            ("ibihumbi_mirongo_ine","4"),
            ("ibihumbi_mirongo_itanu","5"),
            ("ibihumbi_mirongo_itandatatu","6"),
            ("ibihumbi_mirongo_irindwi","7"),
            ("ibihumbi_mirongo_inani","8"),
            ("ibihumbi_mirongo_icyenda","9")
        ])
        hundreds_of_thousands = pynini.string_map([
            ("ibihumbi_ijana","1"),
            ("ibihumbi_magana_abiri","2"),
            ("ibihumbi_magana_atatu","3"),
            ("ibihumbi_magana_ane","4"),
            ("ibihumbi_magana_atanu","5"),
            ("ibihumbi_magana_atandatu","6"),
            ("ibihumbi_magana_arindwi","7"),
            ("ibihumbi_magana_inani","8"),
            ("ibihumbi_magana_cyenda","9")
        ])
        millions = pynini.string_map([
            ("miliyoni","1"),
            ("miliyoni_ebyiri","2"),
            ("miliyoni_eshatu","3"),
            ("miliyoni_enye","4"),
            ("miliyoni_eshanu","5"),
            ("miliyoni_esheshatu","6"),
            ("miliyoni_zirindwi","7"),
            ("miliyoni_umunani","8"),
            ("miliyoni_icyenda","9")
        ])
        tens_of_millions = pynini.string_map([
            ("miliyoni_cumi","1"),
            ("miliyoni_makumyabiri","2"),
            ("miliyoni_mirongo_itatu","3"),
            ("miliyoni_mirongo_ine","4"),
            ("miliyoni_mirongo_itanu","5"),
            ("miliyoni_mirongo_itandatatu","6"),
            ("miliyoni_mirongo_irindwi","7"),
            ("miliyoni_mirongo_inani","8"),
            ("miliyoni_mirongo_icyenda","9")
        ])
        hundreds_of_millions = pynini.string_map([
            ("miliyoni_ijana","1"),
            ("miliyoni_magana_abiri","2"),
            ("miliyoni_magana_atatu","3"),
            ("miliyoni_magana_ane","4"),
            ("miliyoni_magana_atanu","5"),
            ("miliyoni_magana_atandatu","6"),
            ("miliyoni_magana_arindwi","7"),
            ("miliyoni_magana_inani","8"),
            ("miliyoni_magana_cyenda","9")
        ])
        trillions = pynini.string_map([
            ("tiriyoni","1"),
            ("tiriyoni_ebyiri","2"),
            ("tiriyoni_eshatu","3"),
            ("tiriyoni_enye","4"),
            ("tiriyoni_eshanu","5"),
            ("tiriyoni_esheshatu","6"),
            ("tiriyoni_zirindwi","7"),
            ("tiriyoni_umunani","8"),
            ("tiriyoni_icyenda","9")
        ])
        tens_of_trillions = pynini.string_map([
            ("tiriyoni_icumi","1"),
            ("tiriyoni_makumyabiri","2"),
            ("tiriyoni_mirongo_itatu","3"),
            ("tiriyoni_mirongo_ine","4"),
            ("tiriyoni_mirongo_itanu","5"),
            ("tiriyoni_mirongo_itandatatu","6"),
            ("tiriyoni_mirongo_irindwi","7"),
            ("tiriyoni_mirongo_inani","8"),
            ("tiriyoni_mirongo_icyenda","9")
        ])
        hundreds_of_trillions = pynini.string_map([
            ("tiriyoni_ijana","1"),
            ("tiriyoni_magana_abiri","2"),
            ("tiriyoni_magana_atatu","3"),
            ("tiriyoni_magana_ane","4"),
            ("tiriyoni_magana_atanu","5"),
            ("tiriyoni_magana_atandatu","6"),
            ("tiriyoni_magana_arindwi","7"),
            ("tiriyoni_magana_inani","8"),
            ("tiriyoni_magana_cyenda","9")
        ])
        THREE_ZEROS = "000"
        FOUR_ZEROS = "0000"
        FIVE_ZEROS = "00000"
        SIX_ZEROS = "000000"
        SIX_ZEROS = "000000"
        SEVEN_ZEROS = "0000000"
        EIGHT_ZEROS = "00000000"
        NINE_ZEROS = "000000000"

        zero = pynini.string_map([("zeru","0")])
        rewrite_remove_comma_fst = pynini.cdrewrite(pynini.cross(",",""),pynini.union(*"0123456789"),pynini.union(*"0123456789"),NEMO_CHAR.closure())
        single_digits_graph = pynini.invert(digits | zero)
        single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)
        remove_comma = rewrite_remove_comma_fst@single_digits_graph
        
        graph_tens_ends = tens_for_ends +pynutil.delete(" ")+ digits | tens_for_ends+pynutil.insert("0") 
        graph_tens_starts = tens_for_beginnings +pynutil.delete(" ")+ digits | tens_for_beginnings+pynutil.insert("0") 

        graph_tens_for_thousands = tens_for_beginnings +pynutil.delete(" ")+ digits_for_thousands | tens_for_beginnings+pynutil.insert("0") 

        graph_tens_for_millions_trillions = tens_for_beginnings +pynutil.delete(" ")+ digits_millions_trillions \
                                    | tens_for_beginnings+pynutil.insert("0") 
        graph_hundreds = hundreds+pynutil.delete(" ")+graph_tens_ends | hundreds+pynutil.insert("00") \
                                    | hundreds+pynutil.delete(" ")+pynutil.insert("0")+digits
        graph_thousands = thousands+pynutil.delete(" ")+graph_hundreds | thousands+pynutil.insert(THREE_ZEROS) \
                                    | thousands+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_ends \
                                    | thousands+pynutil.delete(" ")+pynutil.insert("00")+digits

        graph_ten_thousand_and_hundreds = ten_thousand +pynutil.insert(THREE_ZEROS) | ten_thousand +pynutil.delete(" ") + graph_hundreds \
                                    | ten_thousand+pynutil.delete(" ") +pynutil.insert("0")+graph_tens_ends \
                                    | ten_thousand+pynutil.delete(" ") +pynutil.insert("00")+digits
        prefix_tens_of_thousands = tens_of_thousands+pynutil.delete(" ") + digits_for_thousands 
        graph_tens_of_thousands = pynutil.add_weight(graph_ten_thousand_and_hundreds, weight=-0.1) \
                                    | prefix_tens_of_thousands+ pynutil.delete(" ")+ graph_hundreds \
                                    | prefix_tens_of_thousands + pynutil.insert(THREE_ZEROS) \
                                    | prefix_tens_of_thousands+pynutil.delete(" ")+pynutil.insert("0")+graph_hundreds \
                                    | prefix_tens_of_thousands+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_ends \
                                    | prefix_tens_of_thousands+pynutil.delete(" ")+pynutil.insert("00")+digits
        
        prefix_hundreds_of_thousands = hundreds_of_thousands+pynutil.delete(" ") + graph_tens_for_thousands 
        graph_hundreds_of_thousands =  hundreds_of_thousands+pynutil.insert(FIVE_ZEROS) \
                                    | prefix_hundreds_of_thousands+pynutil.insert(THREE_ZEROS) \
                                    | prefix_hundreds_of_thousands+pynutil.delete(" ")+graph_hundreds  \
                                    | pynutil.add_weight(prefix_hundreds_of_thousands+pynutil.delete(" ")+pynutil.insert("00")+digits,weight=-0.1) \
                                    | prefix_hundreds_of_thousands+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_for_thousands  
        
        graph_millions = millions +pynutil.delete(" ") + graph_hundreds_of_thousands | millions+pynutil.insert(SIX_ZEROS) \
                                    | millions+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_of_thousands \
                                    | millions+pynutil.delete(" ")+pynutil.insert("00")+graph_thousands \
                                    | millions+pynutil.delete(" ")+pynutil.insert(THREE_ZEROS)+graph_hundreds \
                                    | millions+pynutil.delete(" ")+pynutil.insert(FOUR_ZEROS)+graph_tens_ends \
                                    | millions+pynutil.delete(" ")+pynutil.insert(FIVE_ZEROS)+digits
        
        prefix_tens_of_millions =  tens_of_millions+pynutil.delete(" ") + digits_millions_trillions 
        graph_tens_of_millions = prefix_tens_of_millions +pynutil.delete(" ")+graph_hundreds_of_thousands  \
                                    | prefix_tens_of_millions+pynutil.delete(" ")+pynutil.insert(SIX_ZEROS) \
                                    | prefix_tens_of_millions+pynutil.delete(" ") +pynutil.insert("0")+graph_tens_of_thousands \
                                    | prefix_tens_of_millions+pynutil.delete(" ")+pynutil.insert(THREE_ZEROS)+graph_hundreds \
                                    | prefix_tens_of_millions+pynutil.delete(" ")+pynutil.insert(FOUR_ZEROS)+graph_tens_ends \
                                    | tens_of_millions+pynutil.delete(" ")+pynutil.insert(FIVE_ZEROS)+graph_tens_ends \
                                    | prefix_tens_of_millions+pynutil.delete(" ")+pynutil.insert(FIVE_ZEROS)+digits

        prefix_hundreds_of_millions = hundreds_of_millions+pynutil.delete(" ") +graph_tens_for_millions_trillions
        graph_hundreds_of_millions = prefix_hundreds_of_millions+pynutil.delete(" ")+graph_hundreds_of_thousands \
                                    | prefix_hundreds_of_millions+pynutil.insert(SIX_ZEROS) \
                                    | prefix_hundreds_of_millions+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_of_thousands \
                                    | prefix_hundreds_of_millions+pynutil.delete(" ")+pynutil.insert("00")+graph_thousands \
                                    | prefix_hundreds_of_millions+pynutil.delete(" ")+pynutil.insert(THREE_ZEROS)+graph_hundreds \
                                    | prefix_hundreds_of_millions+pynutil.delete(" ")+pynutil.insert(FOUR_ZEROS)+graph_tens_ends
        
        graph_trillions = trillions+pynutil.delete(" ")+graph_hundreds_of_millions | trillions+pynutil.insert(NINE_ZEROS) \
                                    | trillions+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_of_millions \
                                    | trillions+pynutil.delete(" ")+pynutil.insert("00")+graph_millions \
                                    | trillions+pynutil.delete(" ")+pynutil.insert(THREE_ZEROS)+graph_hundreds_of_thousands \
                                    | trillions+pynutil.delete(" ")+pynutil.insert(FOUR_ZEROS)+graph_tens_of_thousands \
                                    | trillions+pynutil.delete(" ")+pynutil.insert(FIVE_ZEROS)+graph_thousands\
                                    | trillions+pynutil.delete(" ")+pynutil.insert(SIX_ZEROS)+graph_hundreds \
                                    | trillions+pynutil.delete(" ")+pynutil.insert(SEVEN_ZEROS)+graph_tens_ends \
                                    | trillions+pynutil.delete(" ")+pynutil.insert(EIGHT_ZEROS)+digits
        
        prefix_tens_of_trillions =  tens_of_trillions+pynutil.delete(" ") + digits_millions_trillions
        graph_tens_of_trillions = prefix_tens_of_trillions+pynutil.delete(" ")+graph_hundreds_of_millions \
                                    | prefix_tens_of_trillions+pynutil.insert(NINE_ZEROS) \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_of_millions  \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert("00")+graph_millions \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert(THREE_ZEROS)+graph_hundreds_of_thousands \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert(FOUR_ZEROS)+graph_tens_of_thousands \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert(FIVE_ZEROS)+graph_thousands \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert(SIX_ZEROS)+graph_hundreds \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert(SEVEN_ZEROS)+graph_tens_ends \
                                    | prefix_tens_of_trillions+pynutil.delete(" ")+pynutil.insert(EIGHT_ZEROS)+digits 
        
        prefix_hundreds_of_trillions = hundreds_of_trillions+pynutil.delete(" ") +graph_tens_for_millions_trillions
        graph_hundreds_of_trillions = prefix_hundreds_of_trillions+pynutil.delete(" ")+ graph_hundreds_of_millions \
                                    | prefix_hundreds_of_trillions+pynutil.insert(NINE_ZEROS) \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert("0")+graph_tens_of_millions \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert("00")+graph_millions \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert(THREE_ZEROS)+graph_hundreds_of_thousands \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert(FOUR_ZEROS)+graph_tens_of_thousands \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert(FIVE_ZEROS)+graph_thousands \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert(SIX_ZEROS)+graph_hundreds \
                                    | prefix_hundreds_of_trillions+pynutil.delete(" ")+pynutil.insert(SEVEN_ZEROS)+graph_tens_ends
        
        graph_all = graph_hundreds_of_trillions | graph_tens_of_trillions | graph_trillions | graph_hundreds_of_millions | graph_tens_of_millions \
                                    | graph_millions | graph_hundreds_of_thousands | graph_tens_of_thousands \
                                    | graph_thousands | graph_hundreds | pynutil.add_weight(ten, weight=-0.1) \
                                    | graph_tens_starts | digits | pynini.cross("zeru","0") 
                
        inverted_graph_all = pynini.compose(pynini.invert(graph_all),rewrite_add_separator_fst)
        inverted_graph_all = pynini.compose(inverted_graph_all,remove_extra_space_fst)
        inverted_graph_all = pynini.compose(inverted_graph_all,remove_trailing_space_fst)
        inverted_graph_all = pynini.compose(inverted_graph_all,remove_underscore_fst) | pynutil.add_weight(remove_comma, 0.0001)

        inverted_graph_all = inverted_graph_all.optimize()
        final_graph = pynutil.insert("integer: \"") + inverted_graph_all + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph


