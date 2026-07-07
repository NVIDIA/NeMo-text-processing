import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.el.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.el.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.el.taggers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.el.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.el.taggers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.el.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.el.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.el.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.el.taggers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.el.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.el.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.el.taggers.word import WordFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
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
            far_file = os.path.join(cache_dir, f"el_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst(input_case=input_case)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal, input_case=input_case)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal, input_case=input_case)
            decimal_graph = decimal.fst

            fraction_graph = FractionFst(cardinal=cardinal, ordinal=ordinal, input_case=input_case).fst
            measure_graph = MeasureFst(cardinal=cardinal, decimal=decimal, input_case=input_case).fst
            date_graph = DateFst(cardinal=cardinal, input_case=input_case).fst
            word_graph = WordFst().fst
            time_graph = TimeFst(cardinal=cardinal, input_case=input_case).fst
            money_graph = MoneyFst(cardinal=cardinal, input_case=input_case).fst
            whitelist_graph = WhiteListFst(input_file=whitelist, input_case=input_case).fst
            punct_graph = PunctuationFst().fst
            electronic_graph = ElectronicFst(input_case=input_case).fst
            telephone_graph = TelephoneFst(cardinal, input_case=input_case).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 0.9)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.09)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 0.9)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 1.1)
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
