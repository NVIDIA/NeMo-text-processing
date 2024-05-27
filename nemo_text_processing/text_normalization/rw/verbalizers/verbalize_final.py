import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.rw.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
import os

class VerbalizeFinalFst(GraphFst):
    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name="verbalize_final", kind="verbalize")
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"rw_tn_verbalizer.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
            logger.info(f'VerbalizeFinalFst graph was restored from {far_file}.')
        else:
            verbalize = VerbalizeFst().fst
            word = WordFst().fst
		    
            types = verbalize | word
            graph = (
		        pynutil.delete("tokens")
		        + delete_space
		        + pynutil.delete("{")
		        + delete_space
		        + types
		        + delete_space
		        + pynutil.delete("}")
		    )
            graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space


            self.fst = graph

            if far_file:
                generator_main(far_file, {"ALL":self.fst,'REDUP': pynini.accep("REDUP")})
