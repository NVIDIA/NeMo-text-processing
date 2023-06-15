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
from nemo_text_processing.text_normalization.ga.taggers.cardinal import filter_punctuation, make_million
from nemo_text_processing.text_normalization.ga.graph_utils import PREFIX_H, PREFIX_T
from nemo_text_processing.text_normalization.ga.utils import get_abs_path
from pynini.lib import pynutil


def cead_fixup(word):
    word_h = word @ PREFIX_H
    fixup_piece = pynini.cross(word_h, word)
    fixup = pynini.cdrewrite(fixup_piece, "céad" + NEMO_SPACE, "", NEMO_SIGMA)
    return fixup.optimize()


def wrap_word(word: str, deterministic = True, insert_article = False, accept_article = False, insert_word = False, is_date = False, zero_pad = False) -> 'pynini.FstLike':
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
    
    digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
    digit12_nondet = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit12nondet.tsv")))
    digit12_no_endings = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit12.tsv")))
    digit12 = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit12ending.tsv")))
    digit_piece = digit + pynutil.delete("ú")
    digit_piece |= digit12
    if not deterministic:
        digit_piece |= digit12_nondet
        digit_piece |= digit12_no_endings + pynutil.delete("ú")
    if zero_pad:
        digit_graph = pynutil.delete("0") + digit_piece
    else:
        digit_graph = digit_piece

    tens = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/tens.tsv")))
    tens_card = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))
    if not deterministic:
        tens_card |= pynini.cross("4", "ceathracha")
    if is_date:
        tens = pynini.union("2", "3") @ tens
    tens_graph = tens + pynutil.delete("0ú")

    word_h = word @ PREFIX_H
    word_fst = NEMO_SPACE + word_h
    if insert_word:
        word_fst = insert_space + pynutil.insert(word)
    word_inner = word_fst + insert_space

    cead = pynini.string_map([("1d", "céad"), ("1ú", "aonú")])
    dara = pynini.string_map([("2a", "dara"), ("2ú", "dóú")])

    graph = (digit_graph | tens_graph) + word_fst
    graph |= pynutil.delete("1") + digit_piece + word_inner + pynutil.insert("déag")

    if is_date:
        graph |= pynutil.delete("2") + digit_piece + word_inner + pynutil.insert("is fiche")
        graph |= pynutil.delete("3") + cead + word_inner + pynutil.insert("is tríocha")
    else:
        for deich in range(2,10):
            deich = str(deich)
            deich_word = deich @ tens_card
            graph |= pynutil.delete(deich) + digit_piece + word_inner + pynutil.insert("is ") + pynutil.insert(deich_word)

    if article:
        graph = the_article + (graph @ PREFIX_T)
    
    graph_save = graph.write_to_string()
    graph_opt = graph.read_from_string(graph_save)

    return (graph_opt @ cead_fixup(word)).optimize()


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