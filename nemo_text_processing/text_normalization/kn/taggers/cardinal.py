import pynini 
from pynini.lib import pynutil 
 
from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst 
from nemo_text_processing.text_normalization.kn.utils import get_abs_path 
 
 
class CardinalFst(GraphFst): 
    """ 
    Classifies cardinal numbers, e.g.  5  ->  cardinal { integer: "<word>" } 
    """ 
 
    def __init__(self, deterministic: bool = True): 
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic) 
 
        # Load the three data files as transducers  (number -> word) 
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) 
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")) 
        teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")) 
        #hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))
 
        # TODO 1: Combine the three transducers into one grammar so it accepts 
        #         a single digit (1-9), zero (0), OR a two-digit number. 
        #         Hint: the union operator in pynini is  | 
        graph = digit | zero | teens_and_ties            # <-- complete this 
        graph = graph.optimize() 
 
        # TODO 2: Wrap the word in the token field the verbalizer expects: 
        #         produce   integer: "<word>" 
        #         Hint: pynutil.insert("text") writes literal text into the output. 
        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')  
 
        # add_tokens() turns it into:   cardinal { integer: "<word>" } 
        final_graph = self.add_tokens(final_graph) 
        self.fst = final_graph.optimize() 
 