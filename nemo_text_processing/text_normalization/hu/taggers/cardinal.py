# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2022, Jim O'Regan.
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hu.graph_utils import HU_ALPHA
from nemo_text_processing.text_normalization.hu.utils import get_abs_path


def make_million(word: str, hundreds: 'pynini.FstLike', deterministic=False):
    insert_hyphen = pynutil.insert("-")
    # in the non-deterministic case, add an optional space
    if not deterministic:
        insert_hyphen |= pynini.closure(pynutil.insert(" "), 0, 1)

    graph_million = pynutil.add_weight(pynini.cross("001", word), -0.001)
    graph_million |= hundreds + pynutil.insert(word)
    if not deterministic:
        graph_million |= pynutil.add_weight(pynini.cross("001", "egy{word}"), -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", "egy{word} "), -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", "{word} "), -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", " egy{word}"), -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", " egy{word} "), -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", " egy {word} "), -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", " {word} "), -0.001)
    graph_million += insert_hyphen
    graph_million |= pynutil.delete("000")
    return graph_million


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Helper function for parsing number strings. Converts common cardinal strings (groups of three digits delineated by 'cardinal_separator' - see graph_utils)
    and converts to a string of digits:
        "1 000" -> "1000"
        "1.000.000" -> "1000000"
    Args:
        fst: Any pynini.FstLike object. Function composes fst onto string parser fst

    Returns:
        fst: A pynini.FstLike object
    """
    cardinal_separator = pynini.string_map([".", NEMO_SPACE])
    exactly_three_digits = NEMO_DIGIT ** 3  # for blocks of three
    up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)  # for start of string
    up_to_three_digits = up_to_three_digits - "000" - "00" - "0"

    cardinal_string = pynini.closure(
        NEMO_DIGIT, 1
    )  # For string w/o punctuation (used for page numbers, thousand series)

    cardinal_string |= (
        up_to_three_digits
        + pynutil.delete(cardinal_separator)
        + pynini.closure(exactly_three_digits + pynutil.delete(cardinal_separator))
        + exactly_three_digits
    )

    return cardinal_string @ fst


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "ezer" }
        "9999" -> cardinal { integer: "kilencezer-kilencszázkilencvenkilenc" }
        "2000000" -> cardinal { integer: "kétmillió" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        zero = pynini.invert(pynini.string_file(get_abs_path("data/number/zero.tsv")))
        digit = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv")))
        digit_inline = pynini.invert(pynini.string_file(get_abs_path("data/number/digit_inline.tsv")))
        tens = pynini.invert(pynini.string_file(get_abs_path("data/number/tens.tsv")))
        tens_inline = pynini.invert(pynini.string_file(get_abs_path("data/number/tens_inline.tsv")))
        delete_hyphen = pynutil.delete(pynini.closure("-"))
        delete_extra_hyphens = pynini.cross(pynini.closure("-", 1), "-")
        delete_extra_spaces = pynini.cross(pynini.closure(" ", 1), " ")

        # Any single digit
        graph_digit = digit
        self.digit = graph_digit
        graph_zero = zero
        digits_inline_no_one = (NEMO_DIGIT - "1") @ digit_inline
        digits_no_one = (NEMO_DIGIT - "1") @ digit
        if not deterministic:
            graph_digit |= pynini.cross("2", "két")
            digits_inline_no_one |= pynini.cross("2", "kettő")

        insert_hyphen = pynutil.insert("-")
        # in the non-deterministic case, add an optional space
        if not deterministic:
            insert_hyphen |= pynini.closure(pynutil.insert(" "), 0, 1)

        # Any double digit
        graph_tens = (tens_inline + digit) | tens

        self.tens = graph_tens.optimize()

        self.two_digit_non_zero = pynini.union(graph_digit, graph_tens, (pynutil.delete("0") + digit)).optimize()

        base_hundreds = pynini.union(pynini.cross("1", "száz"), digits_inline_no_one + pynutil.insert("száz"))
        if not deterministic:
            base_hundreds |= pynini.cross("1", "egyszáz")
            base_hundreds |= pynini.cross("1", " egyszáz")
            base_hundreds |= pynini.cross("1", "egy száz")
            base_hundreds |= pynini.cross("1", " egy száz")
            base_hundreds |= pynini.cross("1", " száz")
            digits_inline_no_one |= pynutil.insert(" száz")

        hundreds = pynini.union(
            pynini.cross("100", "száz"),
            pynini.cross("1", "száz") + graph_tens,
            digits_inline_no_one + pynini.cross("00", "száz"),
            digits_inline_no_one + pynutil.insert("száz") + graph_tens,
        )
        if not deterministic:
            hundreds |= pynini.union(
                pynini.cross("100", "egyszáz"),
                pynini.cross("1", "egyszáz") + graph_tens,
                pynini.cross("100", " egyszáz"),
                pynini.cross("1", " egyszáz ") + graph_tens,
                pynini.cross("100", "egy száz"),
                pynini.cross("1", "egy száz") + graph_tens,
                pynini.cross("100", " egy száz"),
                pynini.cross("1", " egy száz ") + graph_tens,
                pynini.cross("100", " száz"),
                pynini.cross("1", " száz ") + graph_tens,
            )

        # Three digit strings
        graph_hundreds = base_hundreds + pynini.union(
            pynutil.delete("00"), graph_tens, (pynutil.delete("0") + graph_digit)
        )

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit = (
            graph_hundreds_component_at_least_one_non_zero_digit | graph_tens | graph_digit
        ).optimize()
        # Needed?
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )
        self.hundreds_non_zero_no_one = graph_hundreds_component_at_least_one_non_zero_digit_no_one

        ezer = pynutil.insert("ezer")
        if not deterministic:
            ezer |= pynutil.insert(" ezer")

        ezer1 = ezer
        if not deterministic:
            ezer1 |= pynutil.insert("egyezer")
            ezer1 |= pynutil.insert(" egyezer")
            ezer1 |= pynutil.insert("egy ezer")
            ezer1 |= pynutil.insert(" egy ezer")

        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + ezer
            + insert_hyphen
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
            pynutil.delete("001")
            + ezer1
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + ezer
            + insert_hyphen
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
            pynutil.delete("001")
            + ezer1
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
        )

        self.graph_thousands_component_at_least_one_non_zero_digit = (
            graph_thousands_component_at_least_one_non_zero_digit
        )
        self.graph_thousands_component_at_least_one_non_zero_digit_no_one = (
            graph_thousands_component_at_least_one_non_zero_digit_no_one
        )

        graph_million = make_million("millió", self.hundreds_non_zero_no_one, deterministic)
        graph_milliard = make_million("milliárd", self.hundreds_non_zero_no_one, deterministic)
        graph_billion = make_million("billió", self.hundreds_non_zero_no_one, deterministic)
        graph_billiard = make_million("billiárd", self.hundreds_non_zero_no_one, deterministic)
        graph_trillion = make_million("trillió", self.hundreds_non_zero_no_one, deterministic)
        graph_trilliard = make_million("trilliárd", self.hundreds_non_zero_no_one, deterministic)

        graph = (
            graph_trilliard
            + graph_trillion
            + graph_billiard
            + graph_billion
            + graph_milliard
            + graph_million
            + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))
        )

        clean_output = (
            pynini.cdrewrite(delete_space | delete_hyphen, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space | delete_hyphen, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_extra_hyphens | delete_extra_spaces, "", "", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), HU_ALPHA, HU_ALPHA, NEMO_SIGMA
            )
        ).optimize()

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ clean_output
        )
        zero_space = zero + insert_space
        self.zero_space = zero_space
        self.two_digits_read = pynini.union(
            ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ self.graph_hundreds_component_at_least_one_non_zero_digit,
            zero_space + digit,
        ).optimize()
        self.three_digits_read = pynini.union(
            ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2)) @ self.graph_hundreds_component_at_least_one_non_zero_digit,
            zero_space + ((NEMO_DIGIT ** 2) @ graph_tens),
            zero_space + zero_space + digit,
        ).optimize()
        self.four_digits_read = pynini.union(
            ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 3)) @ self.graph, zero_space + self.three_digits_read
        ).optimize()

        self.graph |= graph_zero

        self.graph_unfiltered = self.graph
        self.graph = filter_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
