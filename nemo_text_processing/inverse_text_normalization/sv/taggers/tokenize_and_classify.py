# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import os

import pynini
from nemo_text_processing.inverse_text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.en.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.sv.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.sv.taggers.cardinal import CardinalFst as TNCardinalTagger
from nemo_text_processing.text_normalization.sv.taggers.date import DateFst as TNDateTagger
from nemo_text_processing.text_normalization.sv.taggers.decimal import DecimalFst as TNDecimalTagger
from nemo_text_processing.text_normalization.sv.taggers.electronic import ElectronicFst as TNElectronicTagger
from nemo_text_processing.text_normalization.sv.taggers.fraction import FractionFst as TNFractionTagger
from nemo_text_processing.text_normalization.sv.taggers.ordinal import OrdinalFst as TNOrdinalTagger
from nemo_text_processing.text_normalization.sv.taggers.telephone import TelephoneFst as TNTelephoneTagger
from nemo_text_processing.text_normalization.sv.verbalizers.electronic import ElectronicFst as TNElectronicVerbalizer
from pynini.lib import pynutil


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != 'None':
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"sv_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            tn_cardinal_tagger = TNCardinalTagger(deterministic=False)
            tn_ordinal_tagger = TNOrdinalTagger(cardinal=tn_cardinal_tagger, deterministic=False)
            tn_date_tagger = TNDateTagger(cardinal=tn_cardinal_tagger, ordinal=tn_ordinal_tagger, deterministic=False)
            tn_decimal_tagger = TNDecimalTagger(cardinal=tn_cardinal_tagger, deterministic=False)
            tn_fraction_tagger = TNFractionTagger(
                cardinal=tn_cardinal_tagger, ordinal=tn_ordinal_tagger, deterministic=True
            )
            tn_electronic_tagger = TNElectronicTagger(deterministic=False)
            tn_electronic_verbalizer = TNElectronicVerbalizer(deterministic=False)
            tn_telephone_tagger = TNTelephoneTagger(deterministic=False)

            cardinal = CardinalFst(tn_cardinal_tagger=tn_cardinal_tagger)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(tn_ordinal=tn_ordinal_tagger)
            ordinal_graph = ordinal.fst
            decimal = DecimalFst(itn_cardinal_tagger=cardinal, tn_decimal_tagger=tn_decimal_tagger)
            decimal_graph = decimal.fst

            fraction = FractionFst(itn_cardinal_tagger=cardinal, tn_fraction_tagger=tn_fraction_tagger)
            fraction_graph = fraction.fst

            date_graph = DateFst(tn_date_tagger=tn_date_tagger).fst
            word_graph = WordFst().fst
            time_graph = TimeFst(tn_cardinal_tagger=tn_cardinal_tagger).fst
            whitelist_graph = WhiteListFst(input_file=whitelist, input_case=input_case).fst
            punct_graph = PunctuationFst().fst
            electronic_graph = ElectronicFst(
                tn_electronic_tagger=tn_electronic_tagger, tn_electronic_verbalizer=tn_electronic_verbalizer
            ).fst
            telephone_graph = TelephoneFst(
                tn_cardinal_tagger=tn_cardinal_tagger, tn_telephone_tagger=tn_telephone_tagger
            ).fst

            classify = (
                pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(whitelist_graph, 1.0)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
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
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
