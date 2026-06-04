import pynini 
from pynini.lib import pynutil 
 
from nemo_text_processing.text_normalization.kn.graph_utils import ( 
    NEMO_NOT_QUOTE, 
    GraphFst, 
    delete_space, 
) 
 
 
class CardinalFst(GraphFst): 
    """ 
    Verbalizes cardinals, e.g.  cardinal { integer: "<word>" }  ->  <word> 
    """ 
 
    def __init__(self, deterministic: bool = True): 
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic) 
 
        # TODO 3: Remove  integer: "  before the word and the closing  "  after it, 
        #         keeping the word itself. 
        #         Hint: pynutil.delete("text") deletes literal text; 
        #               NEMO_NOT_QUOTE matches any character that is not a quote. 
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
 