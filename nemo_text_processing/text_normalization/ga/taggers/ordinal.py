# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2022, Jim O'Regan for Språkbanken Tal
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
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ga.graph_utils import PREFIX_H, PREFIX_T
from nemo_text_processing.text_normalization.ga.taggers.cardinal import filter_punctuation, make_million
from nemo_text_processing.text_normalization.ga.utils import get_abs_path
from pynini.lib import pynutil


def load_digits(deterministic_itn=True, endings=True):
    """
    Everything aside from 'céad' (first) ends with a vowel (or vowel sound)
    so the noun, if it begins with a vowel, needs to have 'h' prefixed, so
    the vowels don't merge.
    TODO: The standard equates 1ú and 2ú to céad and dara respectively; in
    real use, 1d and 2a are used instead. For audio-based normalisation,
    they ought to be accepted.
    """
    digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
    digit1d_nondet = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit1dnondet.tsv")))
    digit1u_nondet = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit1unondet.tsv")))
    digit2_nondet = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit2nondet.tsv")))
    digit1d_ending = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit1dending.tsv")))
    digit1u_ending = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit1uending.tsv")))
    digit2_ending = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit2ending.tsv")))
    if endings:
        digit_h = digit + pynutil.delete("ú") | digit1u_ending | digit2_ending
        digit_d = digit1d_ending

        if not deterministic_itn:
            digit_h |= digit + pynutil.delete("adh")
            digit_h |= digit1u_nondet
            digit_h |= digit2_nondet
            digit_d |= digit1d_nondet
        digit_h_single = digit_h
    else:
        digit_h_single = digit | pynini.cross("2", "dara")
        digit_h = digit | pynini.cross("2", "dóú") | pynini.cross("1", "aonú")
        digit_d = pynini.cross("1", "céad")

    return {"digit_h": digit_h, "digit_h_single": digit_h_single, "digit_d": digit_d}


def wrap_word(
    word: str,
    deterministic=True,
    deterministic_itn=False,
    insert_article=False,
    accept_article=False,
    insert_word=False,
    is_date=False,
    zero_pad=False,
    endings=True,
) -> 'pynini.FstLike':
    if insert_article and accept_article:
        raise ValueError("insert_article and accept_article are mutually exclusive")
    article = False
    if insert_article or accept_article:
        article = True
    the_article = pynini.accep("")
    if insert_article:
        the_article = pynutil.insert("an ")
    if accept_article:
        the_article = pynini.accep("an ")

    delete_u = pynutil.delete("ú")
    if not deterministic_itn:
        delete_u |= pynutil.delete("adh")
    if not endings:
        delete_u = pynini.accep("")

    ordinals = load_digits(deterministic_itn, endings)
    digit_h_single = ordinals["digit_h_single"]
    digit_d = ordinals["digit_d"]
    digit_h = ordinals["digit_h"]

    tens = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/tens.tsv")))
    tens_card = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))
    if not deterministic:
        tens_card |= pynini.cross("4", "ceathracha")
    if is_date:
        tens = pynini.union("2", "3") @ tens
    tens_graph = tens + pynutil.delete("0") + delete_u

    word_h = word @ PREFIX_H
    word_h = NEMO_SPACE + word_h
    word_fst = NEMO_SPACE + word
    if insert_word:
        word_h = insert_space + pynutil.insert(word @ PREFIX_H)
        word_fst = insert_space + pynutil.insert(word)
    word_inner = word_fst + insert_space
    word_h_inner = word_h + insert_space

    if zero_pad:
        graph = (pynutil.delete("0") + digit_h_single | tens_graph) + word_h
        graph |= pynutil.delete("0") + digit_d + word_fst
    else:
        graph = (digit_h_single | tens_graph) + word_h
        graph |= digit_d + word_fst
    graph |= pynini.cross("10", "deichiú") + delete_u + word_h
    graph |= pynutil.delete("1") + digit_h + word_h_inner + pynutil.insert("déag")
    if endings:
        graph |= pynutil.delete("1") + digit_d + word_inner + pynutil.insert("déag")

    if is_date:
        if endings:
            graph |= pynutil.delete("2") + digit_d + word_inner + pynutil.insert("is fiche")
            graph |= pynutil.delete("3") + digit_d + word_inner + pynutil.insert("is tríocha")
        graph |= pynutil.delete("2") + digit_h + word_h_inner + pynutil.insert("is fiche")
        graph |= pynutil.delete("3") + pynini.cross("1", "aonú") + word_h_inner + pynutil.insert("is tríocha")
    else:
        for deich in range(2, 10):
            deich = str(deich)
            deich_word = deich @ tens_card
            if endings:
                graph |= (
                    pynutil.delete(deich) + digit_d + word_inner + pynutil.insert("is ") + pynutil.insert(deich_word)
                )
            graph |= (
                pynutil.delete(deich) + digit_h + word_h_inner + pynutil.insert("is ") + pynutil.insert(deich_word)
            )

    if article:
        graph = the_article + (
            graph @ PREFIX_T @ pynini.cdrewrite(pynini.cross("céad", "chéad"), "[BOS]", "", NEMO_SIGMA)
        )

    return graph.optimize()


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        	"3ú" -> ordinal { integer: "tríú" }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify")
        digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
        ties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/tens.tsv")))
        zero = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/zero.tsv")))
        digit_higher = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit_higher.tsv")))

        final_graph = self.add_tokens(digit)
        self.fst = final_graph.optimize()
