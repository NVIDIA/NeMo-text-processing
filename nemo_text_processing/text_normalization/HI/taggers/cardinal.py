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
from nemo_text_processing.text_normalization.HI.graph_utils import GraphFst, insert_space

 
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

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        graph_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv"))
        graph_thousands = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))
        
        graph_hundred = pynini.cross("१००", "सौ")
        
        
        

        
     
         
        final_graph = graph_digit | graph_zero | graph_teens_and_ties | graph_hundred |  graph_thousands

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        
        self.final_graph = final_graph
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
        
input_text = "१०"                                                                                              
output = apply_fst(input_text,CardinalFst().fst)          # rewrite.rewrites - to see all possible outcomes , rewrite.top_rewrite - shortest pa
print(output)

