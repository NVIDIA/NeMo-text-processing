# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane AI Labs. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.bantu.graph_utils import SINGLE_SPACES
from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile
from nemo_text_processing.inverse_text_normalization.bantu.taggers.tokenize_and_classify import build_classifier
from nemo_text_processing.inverse_text_normalization.lg.settings import load
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_CHAR,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    generator_main,
)
from nemo_text_processing.utils.logging import logger

# "and" before a vowel is written attached to the next word: n'emitwalo (or nemitwalo).
CONTRACTED_AND = ("n'", "n")
# Marks a "ne" made by the split until the kernel has read the sentence; removed before output.
SPLIT = "\x05"


def augmented_scale_words(profile: CardinalProfile):
    """Pairs (augmented, bare) of scale words: a vowel + the bare singular or plural, if the language uses it.

    Luganda: emitwalo/mitwalo, omutwalo/mutwalo, obukadde/bukadde, akakadde/kakadde, ...
    Only forms already in the word lists count, so nothing is invented.
    """
    known = set(profile.number_words())
    for scale in profile.counted_scales:
        known.add(scale.counting_plural)
        known.update(scale.money_only_ones)
        known.update(word for phrase in scale.counting_ones for word in phrase.split())
    return sorted(
        (vowel + bare, bare)
        for scale in profile.counted_scales
        for bare in (scale.singular, scale.plural)
        for vowel in "aeiou"
        if vowel + bare in known
    )


def split_contracted_and(profile: CardinalProfile) -> pynini.Fst:
    """Offers a split of contracted "and" off a scale word, before the kernel reads the sentence.

    yampa emmere n'emitwalo ebiri -> yampa emmere ne mitwalo ebiri -> yampa emmere ne 20,000 UGX,
    and n'omutwalo gumu -> ne mutwalo gumu -> ne 10,000 UGX. The scale word loses its augment
    (emitwalo -> mitwalo, omutwalo -> mutwalo): after "and", an amount is money, and the bare
    form is the one the kernel reads as money (reviewed 2026-11-24). "ne" is the written form
    of the split-off "and". The split is optional and its "ne" is marked, so
    keep_converted_splits can drop it wherever the kernel does not convert the amount after it.
    """
    pairs = [
        (contracted + augmented, "ne" + SPLIT + NEMO_SPACE + bare)
        for augmented, bare in augmented_scale_words(profile)
        for contracted in CONTRACTED_AND
    ]
    left = pynini.union("[BOS]", NEMO_SPACE)
    right = pynini.union("[EOS]", NEMO_SPACE)
    split = pynini.cdrewrite(pynini.string_map(pairs), left, right, NEMO_SIGMA, mode="opt")
    return (SINGLE_SPACES @ split).optimize()


def keep_converted_splits() -> pynini.Fst:
    """Keeps a split only where the kernel converted the amount right after its "ne".

    Anywhere else (no number after it, a held-back word, a malformed or wrongly agreeing
    count) the split reading is rejected and the unsplit words are used, unchanged.
    """
    plain = pynini.difference(NEMO_CHAR, SPLIT)
    converted = pynutil.delete(SPLIT) + '" } tokens { ' + pynini.union("money", "cardinal")
    return pynini.closure(plain | converted).optimize()


def build_luganda_classifier() -> pynini.Fst:
    """The kernel's classifier for Luganda, with contracted "and" split off scale words where that converts."""
    profile = load()
    return (split_contracted_and(profile) @ build_classifier(profile) @ keep_converted_splits()).optimize()


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars for Luganda
    on the shared Bantu kernel. The grammar is built by the kernel from data/ and
    settings.py; see bantu.taggers.tokenize_and_classify.build_classifier. One Luganda
    step is added: contracted "and" is split off a scale word where that gives an amount
    (split_contracted_and, keep_converted_splits).
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        whitelist: not used; Luganda has no whitelist
        overwrite_cache: set to True to overwrite .far files
        input_case: accepting either "lower_cased" or "cased" input. The kernel reads
            lower-cased words and passes other words through, so both build the same grammar.
    """

    def __init__(
        self,
        cache_dir: str = None,
        whitelist: str = None,
        overwrite_cache: bool = False,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"lg_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info("Creating ClassifyFst grammars.")
            self.fst = build_luganda_classifier()
            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logger.info(f"ClassifyFst grammars are saved to {far_file}.")
