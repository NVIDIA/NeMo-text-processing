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
from nemo_text_processing.text_normalization.se.utils import get_abs_path, load_labels, CASE_KEYS, load_case_forms
from pynini.lib import pynutil


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


def load_cased_digits():
    digits_cased = {}
    for key in CASE_KEYS:
        digits_cased[key] = {}
        for label in load_labels(get_abs_path(f"data/numbers/digit_{key}.tsv")):
            digits_cased[key][label[1]] = label[0]
    return digits_cased


def build_cased_fsts(deterministic=True):
    """
    Builds case/number forms (other than nominative singular) for numerals
    See: https://oahpa.no/sme/gramm/logut.eng.html
    """
    digits_cased = load_cased_digits()
    digits_nom = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
    cuodi_cased = load_case_forms(get_abs_path("data/numbers/case_cuodi.tsv"))
    logi_cased = load_case_forms(get_abs_path("data/numbers/case_logi.tsv"))
    duhat_cased = load_case_forms(get_abs_path("data/numbers/case_duhat.tsv"))
    endings_cased = load_case_forms(get_abs_path("data/numbers/digit_case_abbr_suffix.tsv"))

    digits_cased_fst = {}
    for k in digits_cased:
        digits_cased_fst[k] = pynini.string_map((k, v) for k, v in digits_cased[k].items())
    teens_cased_fst = {}
    for k in digits_cased_fst:
        teens_cased_fst[k] = pynutil.delete("1") + digits_cased_fst[k] + pynutil.insert(f"nuppe{logi_cased[k]}")
        if not deterministic:
            teens_cased_fst["ess"] |= pynutil.delete("1") + digits_cased_fst["ess"] + pynutil.insert(f"nuppelogin")
    tens_cased_fst = {}
    # com.sg/loc.pl is different for 'logi'
    for k in digits_cased_fst:
        logi = "logi"
        if k == 'com_sg':
            logi = "logiin"
            ten = digits_cased_fst[k]
        else:
            ten = digits_nom
        # 23 -> guokte/logi/golbma
        tens_cased_fst[k] = digits_cased_fst[k] + pynini.cross("0", logi_cased[k])
        tens_cased_fst[k] |= ten + pynutil.insert(logi) + digits_cased_fst[k]


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "duhat" }
        "2 000 000" -> cardinal { integer: "guoktemiljovnna" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))

        # Any single digit
        graph_digit = digit
        digit_inverse = pynini.invert(digit)
        digits_no_one = (NEMO_DIGIT - "1") @ graph_digit

        graph_zero = zero
        if not deterministic:
            graph_zero |= pynini.cross("0", "nulla")
            graph_digit |= pynini.cross("1", "akta")

        teen = pynutil.delete("1") + digit + pynutil.insert("nuppelohkái")
        teen |= pynini.cross("10", "logi")
        ties = digits_no_one + pynini.cross("0", "logi")
        ties |= digits_no_one + pynutil.insert("logi") + digit

        graph_tens = teen
        graph_ties = ties

        self.tens = graph_tens.optimize()
        self.ties = graph_ties.optimize()

        two_digit_non_zero = pynini.union(graph_tens, graph_ties, (pynutil.delete("0") + graph_digit))
        graph_two_digit_non_zero = pynini.union(graph_digit, two_digit_non_zero)

        self.two_digit_non_zero = graph_two_digit_non_zero.optimize()

        # Three digit strings
        hundreds = digits_no_one + pynutil.insert("čuođi")
        hundreds |= pynini.cross("1", "čuođi")
        if not deterministic:
            hundreds |= pynini.cross("1", "oktačuođi")

        final_hundreds = hundreds + pynini.union(two_digit_non_zero, pynutil.delete("00"))
        graph_hundreds = pynini.union(final_hundreds, graph_two_digit_non_zero)

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + (graph_tens | graph_ties))

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit = (
            graph_hundreds_component_at_least_one_non_zero_digit
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit_no_one = (
            graph_hundreds_component_at_least_one_non_zero_digit_no_one.optimize()
        )

        duhat = pynutil.insert("duhát")
        duhat_cross = pynini.cross("001", "duhát")
        if not deterministic:
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

        graph_million = pynutil.add_weight(pynini.cross("001", "miljon"), -0.001)
        graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("miljovnna")
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "miljun"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "miljovdna"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "milliuvdna"), -0.001)
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("milliuvnna")
            graph_million |= pynutil.add_weight(pynini.cross("001", "miljon "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "miljun "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "miljovdna "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "milliuvdna "), -0.001)
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("miljovnna ")
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "milliuvnna "
            )
        graph_million |= pynutil.delete("000")

        graph_milliard = pynutil.add_weight(pynini.cross("001", "miljárda"), -0.001)
        graph_milliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("miljárdda")
        if not deterministic:
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "miljárda "), -0.001)
            graph_milliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "miljárdda "
            )
        graph_milliard |= pynutil.delete("000")

        graph_billion = pynutil.add_weight(pynini.cross("001", "biljon"), -0.001)
        graph_billion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("biljovnna")
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "biljun"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "biljovdna"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "billiuvdna"), -0.001)
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("billiuvnna")
            graph_million |= pynutil.add_weight(pynini.cross("001", "biljon "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "biljun "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "biljovdna "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "billiuvdna "), -0.001)
            graph_billion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("biljovnna ")
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "billiuvnna "
            )
        graph_billion |= pynutil.delete("000")

        graph_billiard = pynutil.add_weight(pynini.cross("001", "biljárda"), -0.001)
        graph_billiard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("biljárdda")
        if not deterministic:
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "biljárda "), -0.001)
            graph_milliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "biljárdda "
            )
        graph_billiard |= pynutil.delete("000")

        graph_trillion = pynutil.add_weight(pynini.cross("001", "triljon"), -0.001)
        graph_trillion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("triljovnna")
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "triljovdna"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "trilliuvdna"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "triljun"), -0.001)
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "trilliuvnna"
            )
            graph_million |= pynutil.add_weight(pynini.cross("001", "triljon "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "triljun "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "triljovdna "), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "trilliuvdna "), -0.001)
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "triljovnna "
            )
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "trilliuvnna "
            )
        graph_trillion |= pynutil.delete("000")

        graph_trilliard = pynutil.add_weight(pynini.cross("001", "triljárda"), -0.001)
        graph_trilliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("triljárdda")
        if not deterministic:
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "triljárda "), -0.001)
            graph_milliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(
                "triljárdda "
            )
        graph_trilliard |= pynutil.delete("000")

        self.graph_higher = (
            graph_trilliard + graph_trillion + graph_billiard + graph_billion + graph_milliard + graph_million
        )
        graph = self.graph_higher + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))

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

        self.graph = filter_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
