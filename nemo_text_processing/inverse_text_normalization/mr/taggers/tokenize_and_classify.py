import logging
import os

import pynini
from nemo_text_processing.inverse_text_normalization.mr.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.mr.taggers.punctuation import PunctuationFst
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
            far_file = os.path.join(cache_dir, f"mr_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst
            decimal_graph = DecimalFst(cardinal).fst
            time_graph = TimeFst().fst
            date_graph = DateFst(cardinal).fst

            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst
            classify = (
                pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(word_graph, 100)
            )

        punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
        )

        graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
        graph = delete_space + graph + delete_space

        self.fst = graph.optimize()

        if far_file:
            generator_main(far_file, {"tokenize_and_classify": self.fst})
            logging.info(f"ClassifyFst grammars are saved to {far_file}.")

