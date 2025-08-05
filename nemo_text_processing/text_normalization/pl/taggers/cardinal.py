# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2022, 2023 Jim O'Regan for Språkbanken Tal
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
from nemo_text_processing.text_normalization.pl.graph_utils import PL_ALPHA
from nemo_text_processing.text_normalization.pl.utils import adjective_inflection, get_abs_path, load_labels
from nemo_text_processing.text_normalization.pl.taggers.ordinal import complete_paradigm
from pynini.lib import pynutil


CASES = ["nom", "gen", "dat", "acc", "ins", "loc", "voc"]


def make_million(number: str, non_zero_pl: 'pynini.FstLike', non_zero_quant: 'pynini.FstLike', case: str = None, deterministic: bool = True) -> 'pynini.FstLike':
    """
    Helper function for thousands/millions/milliards and higher
    Args:
        number: the string of the number
        non_zero_pl: An fst of digits excluding 0, 1, 5-9, to prefix to plural forms (nom/acc)
        non_zero_quant: An fst of digits excluding 0 and 1-4, to prefix to the quantity forms (nom/acc)
        case: the string of the case (if None, nominative/accusative is presumed)
        deterministic: if True, generate a deterministic fst

    Returns:
        graph: A pynini.FstLike object
    """
    if case is None:
        sg_end = ""
        pl_end = "y"
        quant_end = "ów"
        one = "jeden"
    else:
        SG = {
            "loc": "ie",
            "ins": "em",
            "dat": "owi",
            "gen": "a",
        }
        PL = {
            "loc": "ach",
            "ins": "ami",
            "dat": "om",
            "gen": "ów",
        }
        ONE = {
            "loc": "jednym",
            "ins": "jednym",
            "dat": "jednemu",
            "gen": "jednego",
        }
        sg_end = SG[case]
        pl_end = PL[case]
        one = ONE[case]
        quant_end = pl_end
        if case == "loc" and number.endswith("ard"):
            sg_end = "zie"
    graph = pynutil.add_weight(pynini.cross("001", f"{number}{sg_end}"), -0.001)
    if not deterministic:
        graph |= pynutil.add_weight(pynini.cross("001", f"{one} {number}{sg_end}"), -0.001)
    graph |= non_zero_pl + pynutil.insert(f" {number}{pl_end}")
    # hack for the stem change in tysiąc (1000)
    if number == "tysiąc":
        number = "tysięc"
    graph |= non_zero_quant + pynutil.insert(f" {number}{quant_end}")
    graph |= pynutil.delete("000")
    graph += insert_space
    return graph


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

    cardinal_separator = NEMO_SPACE
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


def make_inflected_graph_dict(file_path: str, cross: str, deterministic=False) -> dict:
    graph_dict = {}
    for line in load_labels(get_abs_path(file_path)):
        key, value = line[0], line[1]
        if key not in graph_dict:
            graph_dict[key] = pynini.cross(cross, value)
        else:
            if not deterministic:
                graph_dict[key] |= pynini.cross(cross, value)
    return graph_dict


def get_nominal_inflections(inflection_file, noun_file):
    output = {}
    inflections = {a[0]: a[1] for a in load_labels(get_abs_path(inflection_file))}
    digit_noun_tsv = load_labels(get_abs_path(noun_file))
    for digit_noun in digit_noun_tsv:
        word = digit_noun[0]
        digit = digit_noun[1]
        lemma_ending = inflections["sg_nom"]
        assert word.endswith(lemma_ending), f"Word {word} does not end with {lemma_ending}"
        stem = word[:-len(lemma_ending)]
        wordforms = {k: stem + v for k, v in inflections.items()}
        output[digit] = wordforms
    return output


def get_nominal_graph(inflection_file, noun_file):
    output = {}
    input = get_nominal_inflections(inflection_file, noun_file)
    for item in input:
        for key in input[item]:
            if not key in output:
                output[key] = pynini.cross(item, input[item][key])
            else:
                output[key] |= pynini.cross(item, input[item][key])
    return output


def get_digit_forms(filepath):
    """
    Returns a dictionary of digit forms for Polish numbers.
    """
    output = {}
    for line in load_labels(get_abs_path(filepath)):
        digit, grammar, form = line[0], line[1], line[2]
        if not digit in output:
            output[digit] = {}
        if grammar not in output[digit]:
            output[digit][grammar] = form
        else:
            if type(output[digit][grammar]) is list:
                output[digit][grammar].append(form)
            else:
                output[digit][grammar] = [output[digit][grammar], form]
    return output


def dict_to_graph(input_dict: dict, deterministic: bool = True) -> dict:
    """
    Converts a nested dictionary of forms to a dict of pynini.FSTs.
    Example input:
        {'2': {'mi_pl_ins': ['form1', 'form2'], 'mi_sg_nom': 'form3'}}
    Output:
        {'2': {'mi_pl_ins': FST, 'mi_sg_nom': FST}}
    """
    graph_dict = {}
    for key, value in input_dict.items():
        graph_dict[key] = {}
        for subkey, subvalue in value.items():
            if isinstance(subvalue, list):
                graph = pynini.cross(key, subvalue[0])
                if not deterministic:
                    for alt in subvalue[1:]:
                        graph |= pynini.cross(key, alt)
            else:
                graph = pynini.cross(key, subvalue)
            graph_dict[key][subkey] = graph
    return graph_dict


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "tysiąc" }
        "2 000 000" -> cardinal { integer: "dwa miliony" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        jeden_all = adjective_inflection("jeden")
        jeden_graph = pynini.cross("1", jeden_all["mi_sg_nom"])
        # in compound numbers, jeden does not inflect
        jeden_only = pynini.cross("1", jeden_all["mi_sg_nom"])
        if not deterministic:
            for key in jeden_all:
                if key == "mi_sg_nom":
                    continue
                jeden_graph |= pynini.cross("1", jeden_all[key])
        complete_paradigm(jeden_all)
        self.jeden_all = {a[0]: pynini.cross("1", a[1]) for a in jeden_all.items()}
        self.zero_all = get_nominal_graph("data/grammar/noun_nt_ro.tsv", "data/numbers/zero.tsv")
        self.zero_sg = {x.replace("sg_", ""): y for x, y in self.zero_all.items() if x.startswith("sg_")}

        dwa_cases = ["mi_pl_nom", "pl_gen", "pl_dat", "mi_pl_nom", "mi_pl_ins", "pl_gen", "mi_pl_nom"]
        pl_cases = ["mi_pl_nom", "pl_gen", "pl_dat", "mi_pl_nom", "pl_ins", "pl_gen", "mi_pl_nom"]
        qnt_cases = ["mi_pl_nom", "pl_gen", "pl_gen", "mi_pl_nom", "pl_ins", "pl_gen", "mi_pl_nom"]

        # jeden (one) does not inflect in compound numbers, so we use the nominative form
        # e.g., https://www.poradnia-jezykowa.uni.lodz.pl/szczegoly/jeden-w-liczebnikach-wielowyrazowych
        # but a lot of people get this wrong, so we also include the inflected forms
        # This is different from Russian; also, jeden in compounds is a quantity, not singular
        jeden_filt = {}
        jeden_compound = {}
        for case in CASES:
            jeden_filt[case] = self.jeden_all[f'mi_sg_{case}']
            jeden_compound[case] = jeden_all[f'mi_sg_nom']
            if not deterministic:
                jeden_compound[case] |= self.jeden_all[f'mi_sg_{case}']

        # 2-4 are plural (5-9 are quantities)
        digit_forms_all = get_digit_forms("data/numbers/digit_forms.tsv")
        digit_graph = dict_to_graph(digit_forms_all, deterministic=deterministic)
        digit_pl = {}
        for idx in range(len(CASES)):
            digit_pl[CASES[idx]] = pynini.union(
                digit_graph["2"][dwa_cases[idx]],
                digit_graph["3"][pl_cases[idx]],
                digit_graph["4"][pl_cases[idx]]
            ).optimize()
        
        digit_qnt = {}


        # zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        # digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
        # teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv")))
        # ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))
        # hundreds = pynini.invert(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")))

        # plural_3digits = NEMO_DIGIT + (NEMO_DIGIT - "1") + pynini.union("2", "3", "4")
        # quantity_3digits = NEMO_DIGIT + pynini.union(
        #     "1" + NEMO_DIGIT,
        #     (NEMO_DIGIT - "1") + pynini.union("0", "5", "6", "7", "8", "9")
        # )

        # # Any single digit
        # graph_digit = digit
        # digits_no_one = (NEMO_DIGIT - "1") @ graph_digit
        # self.digit = graph_digit

        # single_digits_graph = graph_digit | zero
        # self.single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)

        # # Any double digit
        # graph_tens = teen
        # graph_ties = ties
        # if deterministic:
        #     graph_tens |= graph_ties + (pynutil.delete('0') | graph_digit)
        # else:
        #     graph_tens |= pynutil.add_weight(pynini.cross("18", "aderton"), -0.001)
        #     graph_tens |= pynutil.add_weight(
        #         graph_ties + (pynutil.delete('0') | (graph_digit | insert_space + graph_digit)), -0.001
        #     )

        # hundreds = digits_no_one + pynutil.insert("hundra")
        # hundreds |= pynini.cross("1", "hundra")
        # if not deterministic:
        #     hundreds |= pynutil.add_weight(pynini.cross("1", "etthundra"), -0.001)
        #     hundreds |= pynutil.add_weight(digit + pynutil.insert(NEMO_SPACE) + pynutil.insert("hundra"), -0.001)

        # self.tens = graph_tens.optimize()

        # graph_two_digit_non_zero = pynini.union(graph_digit, graph_tens, (pynutil.delete("0") + graph_digit))
        # if not deterministic:
        #     graph_two_digit_non_zero |= pynutil.add_weight(
        #         pynini.union(graph_digit, graph_tens, (pynini.cross("0", NEMO_SPACE) + graph_digit)), -0.001
        #     )

        # self.two_digit_non_zero = graph_two_digit_non_zero.optimize()

        # graph_final_two_digit_non_zero = pynini.union(final_digit, graph_tens, (pynutil.delete("0") + final_digit))
        # if not deterministic:
        #     graph_final_two_digit_non_zero |= pynutil.add_weight(
        #         pynini.union(final_digit, graph_tens, (pynini.cross("0", NEMO_SPACE) + final_digit)), -0.001
        #     )

        # self.final_two_digit_non_zero = graph_final_two_digit_non_zero.optimize()

        # # Three digit strings
        # graph_hundreds = hundreds + pynini.union(pynutil.delete("00"), graph_tens, (pynutil.delete("0") + final_digit))
        # if not deterministic:
        #     graph_hundreds |= pynutil.add_weight(
        #         hundreds
        #         + pynini.union(
        #             pynutil.delete("00"),
        #             (graph_tens | pynutil.insert(NEMO_SPACE) + graph_tens),
        #             (pynini.cross("0", NEMO_SPACE) + final_digit),
        #         ),
        #         -0.001,
        #     )

        # self.hundreds = graph_hundreds.optimize()

        # # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        # graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        # graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
        #     pynutil.delete("00") + graph_digit
        # )

        # graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
        #     pynutil.delete("00") + digits_no_one
        # )
        # self.graph_hundreds_component_at_least_one_non_zero_digit_no_one = (
        #     graph_hundreds_component_at_least_one_non_zero_digit_no_one.optimize()
        # )

        # tusen = pynutil.insert("tusen")
        # etttusen = tusen

        # following_hundred = insert_space + graph_hundreds_component_at_least_one_non_zero_digit
        # if not deterministic:
        #     following_hundred |= graph_hundreds_component_at_least_one_non_zero_digit

        # graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
        #     pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
        #     graph_hundreds_component_at_least_one_non_zero_digit_no_one
        #     + tusen
        #     + (following_hundred | pynutil.delete("000")),
        #     pynini.cross("001", etttusen) + (following_hundred | pynutil.delete("000")),
        # )
        # self.graph_thousands_component_at_least_one_non_zero_digit = (
        #     graph_thousands_component_at_least_one_non_zero_digit.optimize()
        # )

        # graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
        #     pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
        #     graph_hundreds_component_at_least_one_non_zero_digit_no_one
        #     + tusen
        #     + (following_hundred | pynutil.delete("000")),
        #     pynini.cross("001", etttusen) + (following_hundred | pynutil.delete("000")),
        # )
        # self.graph_thousands_component_at_least_one_non_zero_digit_no_one = (
        #     graph_thousands_component_at_least_one_non_zero_digit_no_one.optimize()
        # )

        # non_zero_no_one = graph_hundreds_component_at_least_one_non_zero_digit_no_one
        # graph_million = make_million("milion", non_zero_no_one, deterministic)
        # graph_milliard = make_million("miliard", non_zero_no_one, deterministic)
        # graph_billion = make_million("bilion", non_zero_no_one, deterministic)
        # graph_billiard = make_million("biliard", non_zero_no_one, deterministic)
        # graph_trillion = make_million("trilion", non_zero_no_one, deterministic)
        # graph_trilliard = make_million("triliard", non_zero_no_one, deterministic)

        # graph = (
        #     graph_trilliard
        #     + graph_trillion
        #     + graph_billiard
        #     + graph_billion
        #     + graph_milliard
        #     + graph_million
        #     + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))
        # )

        # self.graph = (
        #     ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
        #     @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
        #     @ NEMO_DIGIT ** 24
        #     @ graph
        #     @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
        #     @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
        #     @ pynini.cdrewrite(
        #         pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), PL_ALPHA, PL_ALPHA, NEMO_SIGMA
        #     )
        # )

        # self.graph_hundreds_component_at_least_one_non_zero_digit = (
        #     pynini.closure(NEMO_DIGIT, 2, 3) | pynini.difference(NEMO_DIGIT, pynini.accep("0"))
        # ) @ self.graph
        # self.graph_hundreds_component_at_least_one_non_zero_digit_en = (
        #     self.graph_hundreds_component_at_least_one_non_zero_digit
        #     @ pynini.cdrewrite(ett_to_en, "", "[EOS]", NEMO_SIGMA)
        # )
        # # For plurals, because the 'one' in 21, etc. still needs to agree
        # self.graph_hundreds_component_at_least_one_non_zero_digit_no_one = (
        #     pynini.project(self.graph_hundreds_component_at_least_one_non_zero_digit, "input") - "1"
        # ) @ self.graph_hundreds_component_at_least_one_non_zero_digit
        # self.graph_hundreds_component_at_least_one_non_zero_digit_no_one_en = (
        #     pynini.project(self.graph_hundreds_component_at_least_one_non_zero_digit_en, "input") - "1"
        # ) @ self.graph_hundreds_component_at_least_one_non_zero_digit_en

        # zero_space = zero + insert_space
        # self.zero_space = zero_space
        # self.three_digits_read = pynini.union(
        #     ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2))
        #     @ self.graph_hundreds_component_at_least_one_non_zero_digit_no_one,
        #     zero_space + ((NEMO_DIGIT ** 2) @ graph_tens),
        #     zero_space + zero_space + digit,
        # )
        # self.three_digits_read_en = pynini.union(
        #     ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2))
        #     @ self.graph_hundreds_component_at_least_one_non_zero_digit_no_one_en,
        #     zero_space + ((NEMO_DIGIT ** 2) @ graph_tens),
        #     zero_space + zero_space + digit,
        # )
        # self.three_digits_read_frac = pynini.union(
        #     ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2))
        #     @ self.graph_hundreds_component_at_least_one_non_zero_digit_no_one,
        #     zero_space + digit + insert_space + digit,
        # )
        # self.three_digits_read_frac_en = pynini.union(
        #     ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2))
        #     @ self.graph_hundreds_component_at_least_one_non_zero_digit_no_one_en,
        #     zero_space + digit + insert_space + digit,
        # )
        # self.two_or_three_digits_read_frac = pynini.union(
        #     ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2))
        #     @ self.graph_hundreds_component_at_least_one_non_zero_digit_no_one,
        #     ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ graph_tens,
        #     zero_space + single_digits_graph + pynini.closure(insert_space + digit, 0, 1),
        #     single_digits_graph + pynini.closure(insert_space + single_digits_graph, 3),
        #     zero_space + zero_space + zero,
        #     single_digits_graph,
        # )
        # self.two_or_three_digits_read_frac_en = pynini.union(
        #     ((NEMO_DIGIT - "0") + (NEMO_DIGIT ** 2))
        #     @ self.graph_hundreds_component_at_least_one_non_zero_digit_no_one_en,
        #     ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ (graph_tens @ pynini.cdrewrite(ett_to_en, "", "[EOS]", NEMO_SIGMA)),
        #     zero_space + single_digits_graph + pynini.closure(insert_space + single_digits_graph, 0, 1),
        #     single_digits_graph + pynini.closure(insert_space + single_digits_graph, 3),
        #     zero_space + zero_space + zero,
        #     single_digits_graph,
        # )
        # self.two_digits_read = pynini.union(((NEMO_DIGIT - "0") + NEMO_DIGIT) @ graph_tens, zero_space + digit)
        # self.two_digits_read_en = pynini.union(
        #     ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ (graph_tens @ pynini.cdrewrite(ett_to_en, "", "[EOS]", NEMO_SIGMA)),
        #     zero_space + digit,
        # )
        # self.any_read_digit = ((NEMO_DIGIT - "0") @ digit) + pynini.closure(insert_space + digit)
        # if not deterministic:
        #     self.three_digits_read |= pynutil.add_weight(digit + insert_space + digit + insert_space + digit, -0.001)
        #     self.three_digits_read |= pynutil.add_weight(
        #         ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ graph_tens + insert_space + digit, -0.001
        #     )
        #     self.three_digits_read |= pynutil.add_weight(
        #         digit + insert_space + ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ graph_tens, -0.001
        #     )
        #     self.two_digits_read |= pynutil.add_weight(digit + insert_space + digit, -0.001)

        # self.graph |= zero

        # self.graph_unfiltered = self.graph
        # self.graph = filter_punctuation(self.graph).optimize()
        # self.graph_en = self.graph @ pynini.cdrewrite(ett_to_en, "", "[EOS]", NEMO_SIGMA)
        # self.graph_no_one = (pynini.project(self.graph, "input") - "1") @ self.graph
        # self.graph_no_one_en = (pynini.project(self.graph_en, "input") - "1") @ self.graph_en

        # joiner_chars = pynini.union("-", "–", "—")
        # joiner = pynini.cross(joiner_chars, " till ")
        # self.range = self.graph + joiner + self.graph
        # if not deterministic:
        #     either_one = self.graph | self.graph_en
        #     self.range = either_one + joiner + either_one

        # optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        # final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        # if not deterministic:
        #     final_graph |= pynutil.add_weight(
        #         optional_minus_graph + pynutil.insert("integer: \"") + self.graph_en + pynutil.insert("\""), -0.001
        #     )
        #     final_graph |= pynutil.add_weight(
        #         pynutil.insert("integer: \"") + self.single_digits_graph + pynutil.insert("\""), -0.001
        #     )

        # final_graph = self.add_tokens(final_graph)
        # self.fst = final_graph.optimize()
