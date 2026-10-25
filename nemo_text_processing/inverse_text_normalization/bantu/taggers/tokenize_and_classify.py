# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile
from nemo_text_processing.inverse_text_normalization.bantu.taggers.sentence import build_tagger
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, generator_main


class ClassifyFst(GraphFst):
    """
    Sentence-level classifier for a Bantu language, configured by a language profile.

    Unlike most languages, the whole sentence is read at once: number words are
    resolved in their context (the counted noun, a currency phrase, a money verb,
    or a join word followed by an ordinary word), and the value is fully resolved
    here. The verbalizer only formats it. Output is the standard token format, e.g.
        "ente bbiri" -> tokens { name: "ente" } tokens { cardinal { integer: "2" } }

    Args:
        profile: the language profile (word lists, agreement, scales, currencies)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, profile: CardinalProfile, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"{profile.language}_itn.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars for {profile.language}.")
            self.fst = build_tagger(profile)
            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
