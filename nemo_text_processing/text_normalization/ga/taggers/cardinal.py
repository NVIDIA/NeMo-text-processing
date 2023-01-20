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
from nemo_text_processing.text_normalization.ga.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.ga.graph_utils import LOWER_LENITION, LOWER_ECLIPSIS
from pynini.lib import pynutil

zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
digit_count = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit_count.tsv")))
teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teens_count.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))


def make_number_form(word: str, deterministic = True, teens = False, tens = False) -> 'pynini.FstLike':
    fst = pynini.accep(word)
    if tens and not teens:
        teens = True
    numbers_len = pynini.string_map([
        ("2", "dhá"),
        ("3", "trí"),
        ("4", "ceithre"),
        ("5", "cúig"),
        ("6", "sé"),
    ])
    numbers_ecl = pynini.string_map([
        ("7", "seacht"),
        ("8", "ocht"),
        ("9", "naoi"),
    ])
    output = pynini.union(
        pynutil.delete("1") + pynutil.insert(fst),
        numbers_len + insert_space + pynutil.insert(fst @ LOWER_LENITION),
        numbers_ecl + insert_space + pynutil.insert(fst @ LOWER_ECLIPSIS)
    )
    if not deterministic:
        output |= pynini.cross("1", "aon") + insert_space + pynutil.insert(fst @ LOWER_LENITION)

    if teens:
        deag = pynini.accep("déag")
        if word[-1] in "aáeéiíoóuú":
            deag = deag @ LOWER_LENITION
        teen_graph = pynutil.delete("1") + output + insert_space + pynutil.insert(deag)
        if not tens:
            output |= teen_graph

    if tens:
        tens_words = load_labels(get_abs_path("data/numbers/tens.tsv"))
        for numword, num in tens_words:
            tmp_graph = pynutil.delete(num) + output + pynutil.insert(" is ") + pynutil.insert(numword)
            output |= tmp_graph
        output |= teen_graph

    return output


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Helper function for parsing number strings. Converts common cardinal strings (groups of three digits delineated by 'cardinal_separator' - see graph_utils)
    and converts to a string of digits:
        "1,000" -> "1000"
        "1,000,000" -> "1000000"
    Args:
        fst: Any pynini.FstLike object. Function composes fst onto string parser fst

    Returns:
        fst: A pynini.FstLike object
    """
    cardinal_separator = pynini.accep(",")
    exactly_three_digits = NEMO_DIGIT ** 3  # for blocks of three
    up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)  # for start of string

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
        "1000" ->  cardinal { integer: "mil" }
        "2.000.000" -> cardinal { integer: "dos millones" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Any single digit
        graph_digit = digit_count
        digits_no_one = (NEMO_DIGIT - "1") @ graph_digit

        # Any double digit
        graph_tens = teen
        graph_tens |= ties + (pynutil.delete('0') | insert_space + graph_digit)
        if not deterministic:
            graph_tens |= ties + (pynutil.delete('0') | (pynutil.insert(" is ") + graph_digit))

        self.tens = graph_tens.optimize()

        self.two_digit_non_zero = pynini.union(
            graph_digit, graph_tens, (pynini.cross("0", NEMO_SPACE) + graph_digit)
        ).optimize()

        # Three digit strings
        hundreds = make_number_form("céad")
        graph_hundreds = hundreds + pynini.union(
            pynutil.delete("00"), (insert_space + graph_tens), (pynini.cross("0", NEMO_SPACE) + graph_digit)
        )

        self.hundreds = graph_hundreds.optimize()
        self.up_to_three_digits = (self.hundreds | graph_tens | graph_digit)
        self.three_digit_non_zero = pynini.union(
            graph_digit,
            self.hundreds,
            graph_tens,
            (pynini.cross("0", NEMO_SPACE) + graph_tens),
            (pynini.cross("00", NEMO_SPACE) + graph_digit)
        ).optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_none_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_none_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        thousands = make_number_form("míle")
        graph_thousands_component_at_least_one_none_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit,
            graph_hundreds_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" mil")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", "mil")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_none_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit_no_one,
            graph_hundreds_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert(" mil")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", "mil")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
        )

        graph_million = pynutil.add_weight(pynini.cross("000001", "un millón"), -0.001)
        graph_million |= graph_thousands_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" millones")
        graph_million |= pynutil.delete("000000")
        graph_million += insert_space

        graph_billion = pynutil.add_weight(pynini.cross("000001", "un billón"), -0.001)
        graph_billion |= graph_thousands_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" billones")
        graph_billion |= pynutil.delete("000000")
        graph_billion += insert_space

        graph_trillion = pynutil.add_weight(pynini.cross("000001", "un trillón"), -0.001)
        graph_trillion |= graph_thousands_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" trillones")
        graph_trillion |= pynutil.delete("000000")
        graph_trillion += insert_space

        graph = (
            graph_trillion
            + graph_billion
            + graph_million
            + (graph_thousands_component_at_least_one_none_zero_digit | pynutil.delete("000000"))
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
        self.graph |= zero

        self.graph = filter_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
