# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, generator_main
from nemo_text_processing.text_normalization.ja.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ja.taggers.date import DateFst
from nemo_text_processing.text_normalization.ja.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ja.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.ja.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.ja.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.ja.taggers.time import TimeFst
from nemo_text_processing.text_normalization.ja.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.ja.taggers.word import WordFst


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(cache_dir, f"zh_tn_{deterministic}_deterministic_{whitelist_file}_tokenize.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            cardinal = CardinalFst(deterministic=deterministic)
            date = DateFst(cardinal=cardinal, deterministic=deterministic)
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            time = TimeFst(cardinal=cardinal, deterministic=deterministic)
            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            whitelist = WhiteListFst(deterministic=deterministic)
            word = WordFst(deterministic=deterministic)
            punctuation = PunctuationFst(deterministic=deterministic)

            classify = pynini.union(
                pynutil.add_weight(date.fst, 1.1),
                pynutil.add_weight(fraction.fst, 1.0),
                pynutil.add_weight(time.fst, 1.1),
                pynutil.add_weight(whitelist.fst, 1.1),
                pynutil.add_weight(cardinal.fst, 1.1),
                pynutil.add_weight(decimal.fst, 3.05),
                pynutil.add_weight(ordinal.fst, 1.1),
                pynutil.add_weight(punctuation.fst, 1.0),
                pynutil.add_weight(word.fst, 100),
            )

            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")
            tagger = pynini.closure(token, 1)

            self.fst = tagger

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
