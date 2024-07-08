# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    GraphFst, 
    NEMO_SIGMA,
    NEMO_HINDI_DIGIT,
)
from pynini.lib import pynutil, rewrite
def apply_fst(text, fst):
  """ Given a string input, returns the output string
  produced by traversing the path with lowest weight.
  If no valid path accepts input string, returns an
  error.
  """
  #try:
     #print(pynini.shortestpath(text @ fst).string())
  #except pynini.FstOpError:
    #print(f"Error: No valid output with given input: '{text}'")
 
class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties.tsv"))
        graph_digit_hundred = pynini.string_file(get_abs_path("data/ordinals/hundred_digit.tsv"))
        graph = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_digit, graph_teens, graph_digit_hundred, pynini.cross("वाँ", "वाँ"), pynini.cross("वां","वां"), pynini.cross("वें", "वें"), pynini.cross("वे", "वे")
        )

        graph_fem_digit = pynini.string_file(get_abs_path("data/ordinals/digit_fem.tsv"))
        graph_fem_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties_fem.tsv"))
        graph_digit_hundred_fem = pynini.string_file(get_abs_path("data/ordinals/hundred_digit_fem.tsv"))
        graph_fem = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_fem_digit, graph_fem_teens, graph_digit_hundred_fem, pynini.cross("वीं", "वीं"), pynini.cross("वी", "वी")
        )
        graph = pynini.compose(graph | graph_fem, (cardinal_graph + pynini.union(pynini.cross("वाँ", "वाँ"), pynini.cross("वां","वां"), pynini.cross("वीं", "वीं"), pynini.cross("वी", "वी"), pynini.cross("वें", "वें"), pynini.cross("वे", "वे") ))).optimize()

#Optimize the following. 
        
        morpho_graph = pynini.cross("वाँ", "\" morphosyntactic_features: \"वाँ\"")
        morpho_graph|= pynini.cross("वां", "\" morphosyntactic_features: \"वां\"")
        morpho_graph|= pynini.cross("वीं", "\" morphosyntactic_features: \"वीं\"")
        morpho_graph|= pynini.cross("वी", "\" morphosyntactic_features: \"वी\"")
        morpho_graph|= pynini.cross("वें", "\" morphosyntactic_features: \"वें\"")
        morpho_graph|= pynini.cross("वे", "\" morphosyntactic_features: \"वे\"")
        
        rule = pynini.cdrewrite(morpho_graph, pynini.closure(NEMO_HINDI_DIGIT), pynini.union("[EOS]"," "), NEMO_SIGMA ) 

        final_graph = (pynutil.insert("integer: \"") + graph@rule )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


