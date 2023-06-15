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
from nemo_text_processing.text_normalization.ga.graph_utils import PREFIX_H, PREFIX_T, bos_or_space, eos_or_space
from nemo_text_processing.text_normalization.ga.utils import get_abs_path
from pynini.lib import pynutil


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
    fixup_piece = "céad " + pynini.cross(word_h, word)
    fixup = pynini.cdrewrite(fixup_piece, "", "", NEMO_SIGMA)
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

    return graph


def wrap_word_wrapper(word: str, deterministic = True, insert_article = False, accept_article = False, insert_word = False, is_date = False, zero_pad = False) -> 'pynini.FstLike':
    graph = wrap_word(word, deterministic, insert_article, accept_article, insert_word, is_date, zero_pad)
    word_h = word @ PREFIX_H
    fixup_piece = "céad " + pynini.cross(word_h, word)
    fixup = pynini.cdrewrite(fixup_piece, "", "", NEMO_SIGMA)
    return (graph @ fixup).optimize()


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

        graph_digit = digit.optimize()
        graph_ties = ties.optimize()
        graph_card_ties = cardinal.ties.optimize()
        graph_card_digit = cardinal.digit.optimize()
        digits_no_one = (NEMO_DIGIT - "1") @ graph_card_digit

        graph_tens_component = graph_teens | graph_card_ties + graph_digit | graph_ties + pynutil.delete('0')
        self.graph_tens_component = graph_tens_component
        graph_tens = graph_tens_component

        digit_or_space = pynini.closure(NEMO_DIGIT | pynini.accep(" "))
        cardinal_format = (NEMO_DIGIT - "0") + pynini.closure(digit_or_space + NEMO_DIGIT, 0, 1)
        a_format = (
            (pynini.closure(cardinal_format + (NEMO_DIGIT - "1"), 0, 1) + pynini.union("1", "2"))
            | (NEMO_DIGIT - "1") + pynini.union("1", "2")
            | pynini.union("1", "2")
        ) + pynutil.delete(pynini.union(":a", ":A"))
        e_format = pynini.closure(
            (NEMO_DIGIT - "1" - "2")
            | (cardinal_format + "1" + NEMO_DIGIT)
            | (cardinal_format + (NEMO_DIGIT - "1") + (NEMO_DIGIT - "1" - "2")),
            1,
        ) + pynutil.delete(pynini.union(":e", ":E"))

        suffixed_ordinal = a_format | e_format
        self.suffixed_ordinal = suffixed_ordinal.optimize()

        bare_hundreds = digits_no_one + pynini.cross("00", "hundrade")

        hundreds = digits_no_one + pynutil.insert("hundra")
        hundreds |= pynini.cross("1", "hundra")

        graph_hundreds = hundreds + pynini.union(graph_tens, (pynutil.delete("0") + graph_digit),)
        graph_hundreds |= bare_hundreds

        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)
        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        self.hundreds = graph_hundreds.optimize()

        tusen = pynutil.insert("tusen")

        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", tusen)
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )
        self.graph_thousands_component_at_least_one_non_zero_digit = (
            graph_thousands_component_at_least_one_non_zero_digit.optimize()
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", tusen)
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )
        self.graph_thousands_component_at_least_one_non_zero_digit_no_one = (
            graph_thousands_component_at_least_one_non_zero_digit_no_one.optimize()
        )

        non_zero_no_one = cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
        graph_million = make_million("miljon", non_zero_no_one, deterministic)
        graph_milliard = make_million("miljard", non_zero_no_one, deterministic)
        graph_billion = make_million("biljon", non_zero_no_one, deterministic)
        graph_billiard = make_million("biljard", non_zero_no_one, deterministic)
        graph_trillion = make_million("triljon", non_zero_no_one, deterministic)
        graph_trilliard = make_million("triljard", non_zero_no_one, deterministic)

        graph = (
            graph_trilliard
            + graph_trillion
            + graph_billiard
            + graph_billion
            + graph_milliard
            + graph_million
            + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))
        )

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA
            )
        )

        cleaned_graph = self.graph
        self.graph |= zero

        self.graph = filter_punctuation(self.graph).optimize()

        self.suffixed_to_words = self.suffixed_ordinal @ self.graph

        self.bare_ordinals = cleaned_graph

        tok_graph = (
            pynutil.insert("integer: \"")
            + (cleaned_graph + pynutil.delete(".") | self.suffixed_to_words)
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(tok_graph)
        self.fst = final_graph.optimize()
