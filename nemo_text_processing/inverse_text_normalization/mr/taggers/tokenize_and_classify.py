import logging
import os

import pynini
from nemo_text_processing.inverse_text_normalization.mr.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from pynini.lib import pynutil

class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment

    Args:

    """
    def __init__(
        self,
        input_case: str = INPUT_LOWER_CASED,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"en_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            classify = pynutil.add_weight(cardinal_graph, 1.1)

        token = pynutil.insert("token { ") + classify + pynutil.insert(" }")

        graph = token
        self.fst = graph.optimize()

        if far_file:
            generator_main(far_file, {"tokenize_and_classify": self.fst})
            logging.info(f"ClassifyFst grammars are saved to {far_file}.")

