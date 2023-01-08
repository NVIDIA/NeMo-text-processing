# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from nemo_text_processing.text_normalization.zh.taggers.char import Char
from nemo_text_processing.text_normalization.zh.taggers.date import Date
from nemo_text_processing.text_normalization.zh.taggers.fraction import Fraction
from nemo_text_processing.text_normalization.zh.taggers.math_symbol import MathSymbol
from nemo_text_processing.text_normalization.zh.taggers.measure import Measure
from nemo_text_processing.text_normalization.zh.taggers.money import Money
from nemo_text_processing.text_normalization.zh.taggers.preprocessor import PreProcessor
from nemo_text_processing.text_normalization.zh.taggers.time import Time
from nemo_text_processing.text_normalization.zh.taggers.whitelist import Whitelist
from pynini.lib import pynutil


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
            far_file = os.path.join(
                cache_dir, f"zh_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            date = Date(deterministic=deterministic)
            cardinal = Cardinal(deterministic=deterministic)
            char = Char(deterministic=deterministic)
            fraction = Fraction(deterministic=deterministic)
            math_symbol = MathSymbol(deterministic=deterministic)
            money = Money(deterministic=deterministic)
            measure = Measure(deterministic=deterministic)
            time = Time(deterministic=deterministic)
            whitelist = Whitelist(deterministic=deterministic)

            classify = pynini.union(
                pynutil.add_weight(date.fst, 1.02),
                pynutil.add_weight(fraction.fst, 1.05),
                pynutil.add_weight(money.fst, 1.05),
                pynutil.add_weight(measure.fst, 1.05),
                pynutil.add_weight(time.fst, 1.05),
                pynutil.add_weight(whitelist.fst, 1.03),
                pynutil.add_weight(cardinal.fst, 1.06),
                pynutil.add_weight(math_symbol.fst, 1.08),
                pynutil.add_weight(char.fst, 100),
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")

            tagger = token.optimize().star

            preprocessor = PreProcessor(remove_interjections=True, fullwidth_to_halfwidth=True,)
            self.fst = preprocessor.fst @ tagger
