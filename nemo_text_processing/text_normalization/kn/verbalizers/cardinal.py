# copyright (c) 2026 NVIDIA Corporation
import pynini 
from pynini.lib import pynutil 
 
from nemo_text_processing.text_normalization.kn.graph_utils import ( 
    NEMO_NOT_QUOTE, 
    GraphFst, 
    delete_space, 
) 
 
 
class CardinalFst(GraphFst): 
    """ 
    Verbalizes cardinals, e.g.  cardinal { integer: "5" }  ->  ಐದು
    """ 
 
    def __init__(self, deterministic: bool = True): 
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic) 
 
        
        graph = ( 
            pynutil.delete("integer:") 
            + delete_space 
            + pynutil.delete('"') 
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"') 
        ) 
 
        # delete_tokens() removes the surrounding  cardinal { ... } 
        delete_tokens = self.delete_tokens(graph) 
        self.fst = delete_tokens.optimize() 
 