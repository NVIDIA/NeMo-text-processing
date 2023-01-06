# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.se.graph_utils import SE_ALPHA
from nemo_text_processing.text_normalization.se.utils import get_abs_path
from pynini.lib import pynutil

zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
ord_digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinal/digit.tsv")))


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Helper function for parsing number strings. Converts common cardinal strings (groups of three digits delineated by space)
    and converts to a string of digits:
        "1 000" -> "1000"
    Args:
        fst: Any pynini.FstLike object. Function composes fst onto string parser fst

    Returns:
        fst: A pynini.FstLike object
    """
    exactly_three_digits = NEMO_DIGIT ** 3  # for blocks of three
    up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)  # for start of string

    cardinal_separator = pynini.union(NEMO_SPACE, ".")
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


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals, e.g.
        "1000" ->  cardinal { integer: "duhat" }
        "2 000 000" -> cardinal { integer: "guoktemiljovnna" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Any single digit
        graph_digit = digit
        digits_no_one = (NEMO_DIGIT - "1") @ graph_digit
        ord_digits_no_one = (NEMO_DIGIT - "1") @ ord_digit

        graph_zero = zero

        teen = pynutil.delete("1") + digit + pynutil.insert("nuppelogát")
        teen |= pynini.cross("10", "logát")
        ties = digits_no_one + pynini.cross("0", "logát")
        ties |= digits_no_one + pynutil.insert("logi") + ord_digit

        graph_tens = teen
        graph_ties = ties

        self.tens = graph_tens.optimize()
        self.ties = graph_ties.optimize()

        two_digit_non_zero = pynini.union(graph_tens, graph_ties, (pynutil.delete("0") + ord_digit))
        graph_two_digit_non_zero = pynini.union(ord_digit, two_digit_non_zero)

        self.two_digit_non_zero = graph_two_digit_non_zero.optimize()

        # Three digit strings
        hundreds = digits_no_one + pynutil.insert("čuođi")
        hundreds |= pynini.cross("1", "čuođi")
        hundreds_ord = digits_no_one + pynutil.insert("čuođát")
        hundreds_ord |= pynini.cross("1", "čuođát")

        final_hundreds = hundreds + two_digit_non_zero
        final_hundreds |= hundreds_ord + pynutil.delete("00")

        graph_hundreds = pynini.union(final_hundreds, graph_two_digit_non_zero)

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + (graph_tens | graph_ties))

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + ord_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + ord_digits_no_one
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit = (
            graph_hundreds_component_at_least_one_non_zero_digit
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit_no_one = (
            graph_hundreds_component_at_least_one_non_zero_digit_no_one.optimize()
        )

        # from here on, construct as though cardinals
        duhat = pynutil.insert("duhát")
        if not deterministic:
            duhat |= pynutil.insert("duhat")
        duhat_cross = pynini.cross("001", "duhát")
        if not deterministic:
            duhat_cross |= pynini.cross("001", "duhat")
            duhat_cross |= pynini.cross("001", "duhat ")
            duhat_cross |= pynini.cross("001", "duhát ")

        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + duhat
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
            duhat_cross + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
        )
        self.graph_thousands_component_at_least_one_non_zero_digit = (
            graph_thousands_component_at_least_one_non_zero_digit
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + duhat
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
            duhat_cross + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
        )

        graph = (
            cardinal.graph_higher
            + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))
        )

        higher_endings = pynini.string_map(
            [
                ("duhát", "duháhat"),
                ("duhat", "duháhat"),
                ("iljárda", "iljárddat"),
                ("iljon", "iljovnnat"),
                ("iljun", "iljovnnat"),
                ("iljovdna", "iljovnnat"),
                ("illiuvdna", "iljovnnat"),
                ("iljovnna", "iljovnnat"),
                ("illiuvnna", "iljovnnat"),
            ]
        )

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), SE_ALPHA, SE_ALPHA, NEMO_SIGMA
            )
        )
        self.graph |= graph_zero
        self.graph @= pynini.cdrewrite(higher_endings, "", "[EOS]", NEMO_SIGMA)

        self.graph_bare_ordinals = filter_punctuation(self.graph).optimize()
        self.graph = (self.graph_bare_ordinals + pynutil.delete(".")).optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
