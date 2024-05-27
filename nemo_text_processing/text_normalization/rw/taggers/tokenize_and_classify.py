from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.rw.taggers.time import TimeFst
from nemo_text_processing.text_normalization.rw.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.taggers.word import WordFst
from nemo_text_processing.text_normalization.rw.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
import pynini
from pynini.lib import pynutil
import os
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)

class ClassifyFst(GraphFst):
    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name='tokenize_and_classify',kind='classify')
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "rw_tn_tokenize_and_classify.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            print("FAR file: ",far_file)
            self.fst = pynini.Far(far_file, mode="r")["TOKENIZE_AND_CLASSIFY"]
        else:
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst
            time_graph = TimeFst().fst
            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst
            whitelist_graph = WhiteListFst().fst
            classify = (
                pynutil.add_weight(time_graph, 1.05)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(word_graph, 1.50)
                | pynutil.add_weight(whitelist_graph,1.01)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token+ pynini.closure(pynutil.insert(" ") + punct)
            )
    
            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space
            self.fst = graph.optimize()
            if far_file:
                print("generating grammar")
                generator_main(far_file, {"TOKENIZE_AND_CLASSIFY":self.fst})
