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
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    MINUS,
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.utils import load_labels
from pynini.lib import pynutil, rewrite

def apply_fst(text, fst):
  """ Given a string input, returns the output string
  produced by traversing the path with lowest weight.
  If no valid path accepts input string, returns an
  error.
  """
  try:
     print(pynini.shortestpath(text @ fst).string())
  except pynini.FstOpError:
    print(f"Error: No valid output with given input: '{text}'")
      
def get_quantity(
    fraction: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike', input_case: str = INPUT_LOWER_CASED
) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or fraction followed by a quantity into a numeral,
    e.g. दस लाख -> integer_part: "१॰" quantity: "लाख"
    e.g. एक बटा पाँच लाख -> numberator: "१" denominator: "५०००००"
    Args:
        fraction: Fraction FST
        cardinal_up_to_hundred: cardinal FST
        input_case: accepting either "lower_cased" or "cased" input.
    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_SIGMA, "0") + pynini.closure(NEMO_SIGMA)
    )
    suffix = pynini.union(
    	"हजार",
	"लाख",
	"करोड़",
	"अरब",
	"खरब",
	"नील",
	"पद्म",
	"शंख",
    )
    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= fraction + delete_extra_space + pynutil.insert("quantity: \"") + suffix + pynutil.insert("\"")
    return res

      
class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        Fraction "/" is determined by "बटा"
            e.g. ऋण एक बटा छब्बीस -> fraction { negative: "true" numerator: "१" denominator: "२६" }
            e.g. छह सौ साठ बटा पाँच सौ तैंतालीस -> fraction { negative: "false" numerator: "६६०" denominator: "५४३" }

 
        The fractional rule assumes that fractions can be pronounced as:
        (a cardinal) + ('बटा') plus (a cardinal, excluding 'zero')
 
    Args:
        cardinal: CardinalFst
        fraction: FractionFst
 
    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
    
        graph_cardinal = cardinal.graph_no_exception
 
        graph_fraction = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_fraction |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_fraction |= pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        graph_fraction |= pynini.string_file(get_abs_path("data/numbers/hundred.tsv")).invert()
        graph_fraction |= pynini.string_file(get_abs_path("data/numbers/thousands.tsv")).invert()
        
        #graph_fraction_exceptions = pynini.string_file(get_abs_path("data/fractions/fraction_exceptions.tsv")).invert()
        
        graph_fraction = pynini.closure(((graph_fraction + delete_space) + graph_fraction) | #graph_fraction_exceptions)
        self.graph = graph_fraction
 
        slash = pynutil.delete("बटा")
 
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ऋण", "\"true\"") + delete_extra_space, 0, 1,
        )
        
        graph_denominator = pynutil.insert("denominator: \"") + graph_fraction + pynutil.insert("\"")
        graph_numerator = pynutil.insert("numerator: \"") + graph_cardinal + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(graph_numerator + delete_extra_space, 0, 1) + slash + delete_extra_space + graph_denominator
        )
        final_graph = optional_graph_negative + final_graph_wo_sign
        
        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, graph_cardinal, input_case=input_case
        )
        
        # accept semiotic spans that start with a capital letter
        self.final_graph_wo_negative |= pynutil.add_weight(
            pynini.compose(TO_LOWER + NEMO_SIGMA, self.final_graph_wo_negative).optimize(), MIN_NEG_WEIGHT
        )
 
        quantity_graph = get_quantity(
            final_graph_wo_sign, graph_cardinal, input_case=input_case
        )
        final_graph |= optional_graph_negative + quantity_graph

        
        if input_case == INPUT_CASED:
            final_graph = capitalized_input_graph(final_graph)
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

#from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
cardinal = CardinalFst()
fraction = FractionFst(cardinal)

input_text = "एक सौ एक बटा चार"
output = apply_fst(input_text, fraction.fst)
print(output)

        
        
        