# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan
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
from nemo_text_processing.text_normalization.ga.graph_utils import (
    GA_ALPHA,
    LOWER_ECLIPSIS,
    LOWER_LENITION,
    PREFIX_H,
    bos_or_space,
    eos_or_space,
)
from nemo_text_processing.text_normalization.ga.utils import get_abs_path, load_labels, load_labels_dict
from pynini.lib import pynutil

zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
zero_count = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero_count.tsv")))
digit_count = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit_count.tsv")))
teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teens_count.tsv")))
teen_noncount = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teens_noncount.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))


def make_number_form(
    word: str, deterministic=True, teens=False, tens=False, higher=False, conjunction=False, numeric=True,
) -> 'pynini.FstLike':
    fst = pynini.accep(word)
    fst_len = fst @ LOWER_LENITION
    fst_ecl = fst @ LOWER_ECLIPSIS
    # If this is a number, make it an insertion
    # Otherwise, we want to word to be retained
    spacer = NEMO_SPACE
    if numeric:
        fst_len = pynutil.insert(fst_len)
        fst_ecl = pynutil.insert(fst_ecl)
        spacer = insert_space
    fst_len_real = fst_len
    # The standard says to inflect "billiún", *but*
    # the standard is of a written-only dialect:
    # it's irrelevant to speech, where inflected forms
    # of billiún clash with milliún
    if word == "billiún":
        billion_fst = pynutil.insert(fst)
        if not numeric:
            billion_fst = fst
        if not deterministic:
            fst_len |= billion_fst
            fst_ecl |= billion_fst
        else:
            fst_len = billion_fst
            fst_ecl = billion_fst

    if tens:
        teens = True

    if higher:
        tens = True
        teens = True

    #  See, e.g.: https://www.lexiconista.com/pdf/Uimhreacha.pdf
    plural_words = load_labels_dict(get_abs_path("data/numbers/plural_nouns.tsv"))
    real_plural = False
    if word in plural_words:
        fst_pl = pynini.accep(plural_words[word])
        fst_len = fst_pl @ PREFIX_H
        fst_ecl = fst_pl @ LOWER_ECLIPSIS
        real_plural = True

    if real_plural:
        fst_len |= pynini.cross(fst, fst_len)
        fst_ecl |= pynini.cross(fst, fst_ecl)

    numbers_len_list = [("2", "dhá"), ("3", "trí"), ("4", "ceithre"), ("5", "cúig"), ("6", "sé")]
    one_form = pynutil.delete("1") + pynutil.insert(fst)
    if real_plural:
        two = pynini.string_map([numbers_len_list[0]]) + spacer + fst_len_real
        one_form = pynini.cross("1", "aon") + spacer + fst_len_real
        numbers_len_list = numbers_len_list[1:]
    numbers_len = pynini.string_map(numbers_len_list)
    numbers_ecl = pynini.string_map([("7", "seacht"), ("8", "ocht"), ("9", "naoi"),])
    output_no_one = pynini.union(numbers_len + spacer + fst_len, numbers_ecl + spacer + fst_ecl)
    single_digit = output_no_one | one_form
    if real_plural:
        single_digit |= two
    if not deterministic:
        lower_len = pynutil.insert(fst @ LOWER_LENITION) if numeric else (fst @ LOWER_LENITION)
        single_digit |= pynini.cross("1", "aon") + spacer + lower_len
    output = single_digit
    if higher:
        output = pynutil.delete("0") + single_digit

    if teens:
        deag = pynini.accep("déag")
        if word[-1] in "aáeéiíoóuú":
            deag = deag @ LOWER_LENITION
        teen_graph = pynutil.delete("1") + output_no_one + insert_space + pynutil.insert(deag)
        if not tens:
            output |= pynini.cross("11", "aon ") + fst_len_real + insert_space + pynutil.insert(deag)
            output |= pynini.cross("10", "deich ") + fst_ecl
            output |= teen_graph

    if tens:
        tens_words = load_labels(get_abs_path("data/numbers/tens.tsv"))
        for numword, num in tens_words:
            tmp_graph = pynutil.delete(num) + single_digit + pynutil.insert(" is ") + pynutil.insert(numword)
            output |= tmp_graph
        output |= pynini.cross("11", "aon ") + fst_len + insert_space + pynutil.insert(deag)
        output |= pynini.cross("10", "deich ") + fst_ecl
        output |= teen_graph

    if conjunction and not deterministic:
        output |= output + pynutil.insert(" is")

    # hundred + 'is' + teens/digit/tens
    # hundred + tens + 'is' + digit
    if higher:
        hundreds = make_number_form("céad")
        hundreds_output = hundreds + pynini.union(
            pynutil.delete("0") + pynutil.insert(" is ") + single_digit,
            pynutil.insert(" is ") + ties + pynutil.delete("0"),
            pynutil.insert(" is ") + output
        )
        output = hundreds_output | (pynutil.delete("0") + output)

    return output


def make_million(word: str, thousands_at_least_one_non_zero_digit_no_one: 'pynini.FstLike') -> 'pynini.FstLike':
    million_like = pynutil.add_weight(pynini.cross("001", word), -0.001)
    million_like |= thousands_at_least_one_non_zero_digit_no_one + pynutil.insert(f" {word}")
    million_like |= pynutil.delete("000")
    million_like += insert_space
    return million_like


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
        "1000" ->  cardinal { integer: "míle" }
        "2,000,000" -> cardinal { integer: "dhá mhilliún" }

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
        base_tens = teen
        base_tens |= ties + (pynutil.delete('0') | insert_space + graph_digit)
        if not deterministic:
            base_tens |= ties + (pynutil.delete('0') | (pynutil.insert(" is ") + graph_digit))

        self.tens = base_tens.optimize()

        graph_tens = teen_noncount
        graph_tens |= ties + (pynutil.delete('0') | insert_space + graph_digit)
        if not deterministic:
            graph_tens |= ties + (pynutil.delete('0') | (pynutil.insert(" is ") + graph_digit))

        self.two_digit_non_zero = pynini.union(
            graph_digit, base_tens, (pynini.cross("0", NEMO_SPACE) + graph_digit)
        ).optimize()

        # Three digit strings
        hundreds = make_number_form("céad")
        graph_hundreds = hundreds + pynini.union(
            pynutil.delete("00"), (insert_space + base_tens), (pynini.cross("0", NEMO_SPACE) + graph_digit)
        )
        if not deterministic:
            graph_hundreds |= hundreds + pynutil.insert(" is") + pynini.cross("0", NEMO_SPACE) + graph_digit
            graph_hundreds |= hundreds + pynutil.insert(" is ") + base_tens
            graph_hundreds = graph_hundreds @ pynini.cdrewrite(
                pynini.cross("is is", "is"), eos_or_space, bos_or_space, NEMO_SIGMA
            )

        self.hundreds = graph_hundreds.optimize()
        self.up_to_three_digits = self.hundreds | base_tens | graph_digit
        self.three_digit_non_zero = pynini.union(
            graph_digit,
            self.hundreds,
            base_tens,
            (pynini.cross("0", NEMO_SPACE) + base_tens),
            (pynini.cross("00", NEMO_SPACE) + graph_digit),
        ).optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        # the different number forms only start to become different at 10,000
        thousands_single_digits = make_number_form("míle", deterministic=deterministic, conjunction=True)
        thousands_single_digits_no_one = (NEMO_DIGIT - "1") @ thousands_single_digits
        self.thousands_single_digits = thousands_single_digits
        # Bunuimhreacha (base numbers)
        thousands_two_digits = make_number_form("míle", deterministic=deterministic, conjunction=True, higher=True)
        self.thousands_two_digits = thousands_two_digits
        thousands_three_digits = (
            graph_hundreds_component_at_least_one_non_zero_digit + insert_space + thousands_two_digits
        )
        self.thousands_three_digits_maol = thousands_three_digits

        # Maoluimhreacha ("bare" numbers)
        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            graph_hundreds_component
            + pynutil.insert(" míle")
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynutil.delete("00")
            + thousands_single_digits
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            graph_hundreds_component
            + pynutil.insert(" míle")
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynutil.delete("00")
            + thousands_single_digits
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        # Bunuimhreacha (base numbers)
        self.million = make_number_form("milliún", deterministic=deterministic, conjunction=True)

        # Maoluimhreacha ("bare" numbers)
        graph_million = make_million("milliún", graph_thousands_component_at_least_one_non_zero_digit_no_one)
        graph_billion = make_million("billiún", graph_thousands_component_at_least_one_non_zero_digit_no_one)
        graph_trillion = make_million("trilliún", graph_thousands_component_at_least_one_non_zero_digit_no_one)
        graph_quadrillion = make_million("cuaidrilliún", graph_thousands_component_at_least_one_non_zero_digit_no_one)
        graph_quintillion = make_million("cuintilliún", graph_thousands_component_at_least_one_non_zero_digit_no_one)
        graph_sextillion = make_million("seisilliún", graph_thousands_component_at_least_one_non_zero_digit_no_one)

        graph = (
            graph_sextillion
            + graph_quintillion
            + graph_quadrillion
            + graph_trillion
            + graph_billion
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
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), GA_ALPHA, GA_ALPHA, NEMO_SIGMA
            )
        )
        self.graph |= zero_count

        self.graph = filter_punctuation(self.graph).optimize()
        self.digit = graph_digit | zero_count
        self.read_digits = self.digit + pynini.closure(pynutil.insert(" ") + self.digit)

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
