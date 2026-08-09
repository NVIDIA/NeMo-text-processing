# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import AbbreviationFst as vAbbreviationFst
from nemo_text_processing.text_normalization.pl.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.pl.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.pl.taggers.date import DateFst
from nemo_text_processing.text_normalization.pl.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.pl.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.pl.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.pl.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.pl.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.pl.taggers.time import TimeFst
from nemo_text_processing.text_normalization.pl.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.pl.verbalizers.cardinal import CardinalFst as vCardinalFst
from nemo_text_processing.text_normalization.pl.verbalizers.date import DateFst as vDateFst
from nemo_text_processing.text_normalization.pl.verbalizers.decimal import DecimalFst as vDecimalFst
from nemo_text_processing.text_normalization.pl.verbalizers.fraction import FractionFst as vFractionFst
from nemo_text_processing.text_normalization.pl.verbalizers.measure import MeasureFst as vMeasureFst
from nemo_text_processing.text_normalization.pl.verbalizers.ordinal import OrdinalFst as vOrdinalFst
from nemo_text_processing.text_normalization.pl.verbalizers.roman import RomanFst as vRomanFst
from nemo_text_processing.text_normalization.pl.verbalizers.time import TimeFst as vTimeFst


class ClassifyFst(GraphFst):
    """Composes Polish classification and verbalization for audio-based TN."""

    def __init__(
        self,
        input_case: str,
        deterministic: bool = False,
        cache_dir: str = None,
        overwrite_cache: bool = True,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"_{input_case}_pl_tn_{deterministic}_with_audio.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            cardinal = CardinalFst(deterministic=deterministic)
            ordinal = OrdinalFst(deterministic=deterministic)
            roman = RomanFst(ordinal, deterministic=deterministic)
            date = DateFst(cardinal, ordinal, deterministic=deterministic)
            decimal = DecimalFst(cardinal, deterministic=deterministic)
            fraction = FractionFst(cardinal, ordinal, deterministic=deterministic)
            measure = MeasureFst(cardinal, ordinal, deterministic=deterministic)
            time = TimeFst(cardinal, ordinal, deterministic=deterministic)
            whitelist_graph = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist)
            v_cardinal = vCardinalFst(deterministic=deterministic)
            v_ordinal = vOrdinalFst(deterministic=deterministic)
            v_roman = vRomanFst(deterministic=deterministic)
            v_date = vDateFst(deterministic=deterministic)
            v_decimal = vDecimalFst(deterministic=deterministic)
            v_fraction = vFractionFst(deterministic=deterministic)
            v_measure = vMeasureFst(deterministic=deterministic)
            v_time = vTimeFst(deterministic=deterministic)
            word = pynini.closure(NEMO_NOT_SPACE, 1)
            punctuation = PunctuationFst(deterministic=True).graph

            sem_w = 1
            word_w = 100
            punct_w = 2
            classify_and_verbalize = (
                pynutil.add_weight(whitelist_graph.graph, sem_w)
                | pynutil.add_weight(pynini.compose(roman.fst, v_roman.fst), sem_w)
                | pynutil.add_weight(pynini.compose(date.fst, v_date.fst), sem_w)
                | pynutil.add_weight(pynini.compose(decimal.fst, v_decimal.fst), sem_w)
                | pynutil.add_weight(pynini.compose(fraction.fst, v_fraction.fst), sem_w)
                | pynutil.add_weight(pynini.compose(measure.fst, v_measure.fst), sem_w)
                | pynutil.add_weight(pynini.compose(time.fst, v_time.fst), sem_w)
                | pynutil.add_weight(pynini.compose(cardinal.fst, v_cardinal.fst), sem_w)
                | pynutil.add_weight(pynini.compose(ordinal.fst, v_ordinal.fst), sem_w)
                | pynutil.add_weight(word, word_w)
            ).optimize()
            if not deterministic:
                abbreviation = AbbreviationFst(whitelist=whitelist_graph, deterministic=False)
                v_abbreviation = vAbbreviationFst(deterministic=False)
                classify_and_verbalize |= pynutil.add_weight(
                    pynini.compose(abbreviation.fst, v_abbreviation.fst), word_w
                )
            punct_only = pynutil.add_weight(punctuation, punct_w)
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | pynutil.insert(" ") + punct_only,
                1,
            )
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + classify_and_verbalize
                + pynini.closure(pynutil.insert(" ") + punct)
            )
            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | pynutil.insert(" ") + punct + pynutil.insert(" ")
                )
                + token_plus_punct
            )
            graph |= punct_only + pynini.closure(punct)
            graph = delete_space + graph + delete_space
            remove_extra_spaces = pynini.closure(NEMO_NOT_SPACE, 1) + pynini.closure(
                delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1)
            )
            remove_extra_spaces |= (
                pynini.closure(pynutil.delete(" "), 1)
                + pynini.closure(NEMO_NOT_SPACE, 1)
                + pynini.closure(delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1))
            )
            self.fst = pynini.compose(graph.optimize(), remove_extra_spaces).optimize()
            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})

        no_digits = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_DIGIT))
        self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()

    def lattice(self, text: str) -> 'pynini.Fst':
        lattice = pynini.compose(pynini.accep(pynini.escape(text)), self.fst)
        if lattice.start() == pynini.NO_STATE_ID:
            raise ValueError(f"Polish TN failed for input: {text}")
        return lattice

    def normalize(self, text: str, lm: Optional['pynini.FstLike'] = None) -> str:
        lattice = self.lattice(text)
        if lm is not None:
            lattice = pynini.compose(lattice, lm)
            if lattice.start() == pynini.NO_STATE_ID:
                raise ValueError("The language model rejected every Polish TN path")
        best = pynini.shortestpath(lattice, nshortest=1, unique=True).project("output")
        return best.string()
