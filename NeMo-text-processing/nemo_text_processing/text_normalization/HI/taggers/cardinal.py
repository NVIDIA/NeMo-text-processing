# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.HI.utils import get_abs_path, apply_fst
from nemo_text_processing.text_normalization.HI.graph_utils import GraphFst, insert_space, delete_space

#def del_zero(n=1):                                         #prevents us from writing the function multiple times 
    #return pynutil.delete("०") ** n 
 
class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -२३ -> cardinal { negative: "true"  integer: "तेइस" } }
 s
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv"))
        thousand = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))
    
        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            insert_space = pynutil.insert(" ")
            zero = pynutil.delete("०")
            del_zero = pynini.closure(zero, zeros_counts, zeros_counts)
            return digit_graph + del_zero + suffix 
        

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            insert_space = pynutil.insert(" ")
            zero = pynutil.delete("०")
            del_zero = pynini.closure(zero, zeros_counts, zeros_counts)
            #graph = digit_graph + del_zero + suffix
            graph = digit_graph + suffix + del_zero + insert_space + sub_graph 
            return graph

        
       
        #Hundred graph
        suffix_hundreds = pynutil.insert(" सौ")
        graph_hundreds = create_graph_suffix(digit, suffix_hundreds, 2)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 1, digit)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 0, teens_and_ties)
        graph_hundreds.optimize()
       
        
        #Thousands and Ten thousands graph 
        suffix_thousands = pynutil.insert(" हज़ार")
        graph_thousands = create_graph_suffix(digit, suffix_thousands, 3)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 2, digit)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 1, teens_and_ties)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 0, graph_hundreds)
        graph_thousands.optimize()
        
        graph_ten_thousands = create_graph_suffix(teens_and_ties, suffix_thousands, 3)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 2, digit)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 1, teens_and_ties)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 0, graph_hundreds)
        graph_ten_thousands.optimize()
    
        #Lakhs graph and ten lakhs graph
        suffix_lakhs = pynutil.insert(" लाख")
        graph_lakhs = create_graph_suffix(digit, suffix_lakhs, 5)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 4, digit)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 3, teens_and_ties) 
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 2, graph_hundreds)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 0, graph_ten_thousands)
        graph_lakhs.optimize()
        
        graph_ten_lakhs = create_graph_suffix(teens_and_ties, suffix_lakhs, 5)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 4, digit)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 3, teens_and_ties)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 2, graph_hundreds)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 0, graph_ten_thousands)
        graph_ten_lakhs.optimize()

        #Crores graph 
        suffix_crores = pynutil.insert(" करोड़")
        graph_crores = create_graph_suffix(digit, suffix_crores, 7) 
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 6, digit)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 5, teens_and_ties)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 4, graph_hundreds)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 0, graph_ten_lakhs)
        graph_crores.optimize()
        
        #Ten crores graph 
        suffix_ten_crores = pynutil.insert(" दस करोड़")
        graph_ten_crores = create_larger_number_graph(teens_and_ties, suffix_crores, 7, graph_lakhs) 
        graph_ten_crores.optimize()
        
        #Arabs graph
        suffix_arabs = pynutil.insert(" अरब")
        graph_arabs = create_larger_number_graph(digit, suffix_arabs, 9, graph_crores) 
        graph_arabs.optimize()
        
        #Ten arabs graph
        suffix_ten_arabs = pynutil.insert(" दस अरब")
        graph_ten_arabs = create_larger_number_graph(teens_and_ties, suffix_arabs, 9, graph_crores) 
        graph_ten_arabs.optimize()
            
        #Kharabs graph
        suffix_kharabs = pynutil.insert(" खरब") 
        graph_kharabs = create_larger_number_graph(digit, suffix_kharabs, 11, graph_arabs)
        graph_kharabs.optimize()
        
        #Ten kharabs graph
        suffix_ten_kharabs = pynutil.insert(" दस खरब") 
        graph_ten_kharabs = create_larger_number_graph(teens_and_ties, suffix_kharabs, 11, graph_arabs)
        graph_ten_kharabs.optimize()
        
        #Nils graph
        suffix_nils = pynutil.insert(" नील")
        graph_nils =  create_larger_number_graph(digit, suffix_nils, 13, graph_kharabs)
        graph_nils.optimize()
        
        #Ten nils graph
        suffix_ten_nils = pynutil.insert(" दस नील")
        graph_ten_nils =  create_larger_number_graph(teens_and_ties, suffix_nils, 13, graph_kharabs)
        graph_ten_nils.optimize()
        
        #Padmas graph 
        suffix_padmas = pynutil.insert(" पद्म")
        graph_padmas = create_larger_number_graph(digit, suffix_padmas, 15, graph_nils)
        graph_padmas.optimize()
        
        #Ten padmas graph 
        suffix_ten_padmas = pynutil.insert(" दस पद्म")
        graph_ten_padmas = create_larger_number_graph(teens_and_ties, suffix_padmas, 15, graph_nils)
        graph_ten_padmas.optimize()
        
        #Shankhs graph
        suffix_shankhs = pynutil.insert(" शंख")
        graph_shankhs =  create_larger_number_graph(digit, suffix_shankhs, 17, graph_padmas)   
        graph_shankhs.optimize()
        
        #Ten shankhs graph
        suffix_ten_shankhs = pynutil.insert(" दस शंख")
        graph_ten_shankhs =  create_larger_number_graph(teens_and_ties, suffix_shankhs, 17, graph_padmas)   
        graph_ten_shankhs.optimize()  

        #shorter_numbers = graph_thousands | pynutil.add_weight(graph_hundreds, 0.01)
        #short_numbers = graph_ten_thousands | pynutil.add_weight(graph_hundreds, 0.01)
        #long_numbers = graph_lakhs | pynutil.add_weight(graph_ten_thousands, 0.1) | pynutil.add_weight(graph_thousands, 0.01) | pynutil.add_weight(graph_hundreds, 0.001)
        
        #long_numbers =  graph_crores | pynutil.add_weight(graph_lakhs, 0.01) | pynutil.add_weight(graph_ten_lakhs, 0.01) | pynutil.add_weight(graph_thousands, 0.01) | pynutil.add_weight(graph_ten_thousands, 0.01) | pynutil.add_weight(graph_hundreds, 0.001) 
            
        long_numbers = (
            pynutil.add_weight(graph_crores, 1.0) | 
            pynutil.add_weight(graph_ten_lakhs, 0.1) | 
            pynutil.add_weight(graph_lakhs, 0.01) | 
            pynutil.add_weight(graph_ten_thousands, 0.001) | 
            pynutil.add_weight(graph_thousands, 0.0001) | 
            pynutil.add_weight(graph_hundreds, 0.00001) 
        )
    
        final_graph = digit | zero | teens_and_ties | long_numbers | graph_hundreds | graph_thousands | graph_ten_thousands | graph_lakhs | graph_ten_lakhs | graph_crores | graph_ten_crores | graph_arabs | graph_ten_arabs | graph_kharabs | graph_ten_kharabs | graph_nils | graph_ten_nils | graph_padmas | graph_ten_padmas | graph_shankhs | graph_ten_shankhs 
        
        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        
        self.final_graph = final_graph
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
        
input_text = "१११०००११"                                                                                              
output = apply_fst(input_text, CardinalFst().fst)          # rewrite.rewrites - to see all possible outcomes , rewrite.top_rewrite - shortest pa
print(output)

