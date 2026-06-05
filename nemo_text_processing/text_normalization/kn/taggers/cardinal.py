# copyright (c) 2026 NVIDIA Corporation
import pynini 
from pynini.lib import pynutil 
 
from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst 
from nemo_text_processing.text_normalization.kn.utils import get_abs_path 
 
 
class CardinalFst(GraphFst): 
    """ 
    Classifies cardinal numbers, e.g.  5  ->  cardinal { integer: "ಐದು" } 
    """ 
 
    def __init__(self, deterministic: bool = True): 
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic) 
 
        # Load the three data files as transducers  (number -> word) 
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) 
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")) 
        teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")) 
        #hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))
 
        
        graph = digit | zero | teens_and_ties             # <-- complete this 
        graph = graph.optimize() 
 
        
        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')  
 
        # add_tokens() turns it into:   cardinal { integer: "<word>" } 
        final_graph = self.add_tokens(final_graph) 
        self.fst = final_graph.optimize() 
 