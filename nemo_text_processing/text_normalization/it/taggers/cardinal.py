# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import cardinal_separator
from nemo_text_processing.text_normalization.it.utils import get_abs_path

zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv")))
tens = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))
tens_one = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens_one.tsv")))
tens_eight = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens_eight.tsv")))
hundreds = pynini.invert(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")))


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
    '''
    Finite state transducer for classifying cardinals in Italian, e.g.
        "1000" ->  cardinal { integer: "mille" }
        "2.000.000" -> cardinal { integer: "due milioni" }
    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    '''

    def __init__(self, deterministic: bool = True):
        super().__init__(name='cardinal', kind='classify', deterministic=deterministic)

        # single digit
        graph_digit = digit
        digits_no_one = (NEMO_DIGIT - "1") @ graph_digit

        # double digit
        graph_tens = teen
        graph_tens |= tens + (pynutil.delete('0') | graph_digit)
        graph_tens |= pynutil.add_weight(tens_one, -0.01)
        graph_tens |= pynutil.add_weight(tens_eight, -0.01)

        self.tens = graph_tens.optimize()

        self.two_digit_no_zero = pynini.union(
            graph_digit, graph_tens, (pynini.cross('0', NEMO_SPACE) + graph_digit)
        ).optimize()

        # three digit
        graph_hundreds = hundreds + pynini.union(
            pynutil.delete('00'), insert_space + graph_tens, (pynini.cross("0", NEMO_SPACE) + graph_digit)
        )
        graph_hundreds |= (
            pynini.cross('1', 'cento')
            + insert_space
            + pynini.union(graph_tens, pynutil.delete("0") + graph_digit, pynutil.delete("00"))
        )

        self.hundreds = graph_hundreds.optimize()

        # three digit leading zeros
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_none_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )

        graph_hundreds_component_at_least_one_none_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        # thousands
        graph_thousands_component_at_least_one_none_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit,
            graph_hundreds_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert("mila")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", "mille")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_none_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit_no_one,
            graph_hundreds_component_at_least_one_none_zero_digit_no_one
            + pynutil.insert("mila")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", "mille")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
        )

        # higher
        graph_million = pynutil.add_weight(pynini.cross("000001", "un milione"), -0.001)
        graph_million |= graph_thousands_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" milioni")
        graph_million |= pynutil.delete("000000")
        graph_million += insert_space

        graph_billion = pynutil.add_weight(pynini.cross("000001", "un miliardo"), -0.001)
        graph_billion |= graph_thousands_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" miliardi")
        graph_billion |= pynutil.delete("000000")
        graph_billion += insert_space

        graph_trillion = pynutil.add_weight(pynini.cross("000001", "un trilione"), -0.001)
        graph_trillion |= graph_thousands_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" trilioni")
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
