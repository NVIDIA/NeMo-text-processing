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
import copy

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
from nemo_text_processing.text_normalization.se.graph_utils import SE_ALPHA, make_spacer
from nemo_text_processing.text_normalization.se.utils import CASE_KEYS, get_abs_path, load_case_forms, load_labels
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


def load_cased_digits(bare=True):
    digits_cased = {}
    for key in CASE_KEYS:
        if bare:
            fkey = f"bare_{key}"
        else:
            fkey = key
        digits_cased[key] = {}
        for label in load_labels(get_abs_path(f"data/numbers/digit_{fkey}.tsv")):
            digits_cased[key][label[1]] = label[0]
    digits_cased["nom_sg"] = {}
    for label in load_labels(get_abs_path(f"data/numbers/digit.tsv")):
        digits_cased["nom_sg"][label[1]] = label[0]
    return digits_cased


def build_cased_number_fsts(deterministic=True):
    """
    Builds case/number forms (other than nominative singular) for numerals
    See: https://oahpa.no/sme/gramm/logut.eng.html
    for teens and tens; for longer numbers Nickel and Sammallahti (2011) say
    'i lengre tallord bøyes bare enere' ('in longer number words only ones are inflected')
    """
    digits_nom = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
    digits_nom_no_one = (NEMO_DIGIT - "1") @ digits_nom
    cuodi_cased = load_case_forms(get_abs_path("data/numbers/case_cuodi.tsv"), True)
    logi_cased = load_case_forms(get_abs_path("data/numbers/case_logi.tsv"), True)
    duhat_cased = load_case_forms(get_abs_path("data/numbers/case_duhat.tsv"), True)
    nolla_cased = load_case_forms(get_abs_path("data/numbers/case_nolla.tsv"), True)
    endings_cased = load_case_forms(get_abs_path("data/numbers/digit_case_abbr_suffix.tsv"))
    spacer = make_spacer(deterministic)

    if not deterministic:
        digits_nom |= pynini.cross("1", "akta")

    # digits
    def get_digit_cased_fst(bare=True, deterministic=deterministic):
        digits_cased = load_cased_digits(bare)

        digits_cased_fst = {}
        for k in digits_cased:
            digits_cased_fst[k] = pynini.string_map((k, v) for k, v in digits_cased[k].items())
            if not deterministic:
                if k == "nom_sg":
                    digits_cased_fst[k] |= pynini.cross("1", "akta")
                elif k == "gen_sg" and bare:
                    digits_cased_fst[k] |= pynini.cross("2", "guovtti")
        return digits_cased_fst

    digits_bare_cased_fst = get_digit_cased_fst(bare=True)
    digits_cased_fst = get_digit_cased_fst(bare=False)

    # for hundreds, thousands, etc.
    digits_nom_prefix = (NEMO_DIGIT - "1") @ digits_nom
    digits_nom_prefix |= pynutil.delete("1")
    if not deterministic:
        digits_nom_prefix |= pynini.cross("1", "okta")
        digits_nom_prefix |= pynini.cross("1", "akta")

    digits_cased_fst_pfx = {}
    digits_cased_fst_pfx["nom_sg"] = digits_nom_prefix
    for k in digits_cased_fst:
        if k not in digits_cased_fst_pfx:
            digits_cased_fst_pfx[k] = digits_cased_fst[k]

    # zero
    nolla_cased_fst = {}
    for k in nolla_cased:
        nolla_cased_fst[k] = pynini.cross("0", nolla_cased[k])

    # teens
    teens_cased_fst = {}
    nuppelogin = pynutil.insert("nuppelogin")
    for k in digits_cased_fst:
        assert "nom_sg" in digits_cased_fst
        assert "nom_sg" in logi_cased
        if k == "nom_sg":
            teens_cased_fst[k] = pynutil.delete("1") + digits_cased_fst[k] + pynutil.insert("nuppelohkái")
        else:
            teens_cased_fst[k] = pynutil.delete("1") + digits_cased_fst[k] + pynutil.insert(f"nuppe{logi_cased[k]}")
        if not deterministic:
            if k in ["nom_pl", "gen_pl", "acc_pl", "loc_pl"]:
                dbc = digits_bare_cased_fst[k]
                teens_cased_fst[k] = pynutil.delete("1") + dbc + pynutil.insert(f"nuppe{logi_cased[k]}")
            if k == "ess":
                teens_cased_fst["ess"] |= pynutil.delete("1") + digits_cased_fst["ess"] + nuppelogin
                teens_cased_fst["ess"] |= pynutil.delete("1") + digits_cased_fst["nom_pl"] + nuppelogin

    # tens
    tens_cased_fst = {}
    # com.sg/loc.pl is different for 'logi'
    for k in digits_cased_fst:
        logi = "logi"
        digit_cased_no_one = (NEMO_DIGIT - "1") @ digits_bare_cased_fst[k]
        digit_nom_no_one = (NEMO_DIGIT - "1") @ digits_bare_cased_fst["nom_sg"]
        if k == 'com_sg':
            logi = "logiin"
            ten = digit_cased_no_one
        else:
            ten = digits_nom_no_one
        # 20 -> guvttiin/logiin
        tens_cased_fst[k] = digit_cased_no_one + spacer + pynini.cross("0", logi_cased[k])
        tens_cased_fst[k] |= pynini.cross("10", logi_cased[k])
        # e.g.: https://gtweb.uit.no/cgi-bin/smi/smi.cgi?text=vihttalogi&pos=Any&mode=full&lang=sme&plang=eng&action=paradigm
        if not deterministic:
            if k in ["nom_pl", "gen_pl", "loc_pl", "ess"]:
                tens_cased_fst[k] |= digit_nom_no_one + spacer + pynini.cross("0", logi_cased[k])
            if k == "ess":
                digit_nom_pl = (NEMO_DIGIT - "1") @ digits_cased_fst["nom_pl"]
                tens_cased_fst[k] |= digit_nom_pl + spacer + pynini.cross("0", "login")

                tens_cased_fst[k] |= digit_nom_no_one + spacer + pynini.cross("0", "login")
            if k == "nom_sg":
                tens_cased_fst[k] |= digit_nom_no_one + spacer + pynini.cross("0", "lohki")
            if k == "com_pl":
                digit_gen_pl = (NEMO_DIGIT - "1") @ digits_cased_fst["gen_pl"]
                tens_cased_fst[k] |= digit_gen_pl + spacer + pynini.cross("0", logi_cased[k])
        # 23 -> guvttiin/logiin/golmmain
        tens_cased_fst[k] |= ten + spacer + pynutil.insert(logi) + spacer + digits_bare_cased_fst[k]

    # two digits
    two_digit_cased_fsts = {}
    two_digit_cased_fsts_sfx = {}
    two_digits_fst = None
    for k in digits_cased_fst:
        two_digit_cased_fsts[k] = (
            tens_cased_fst[k] | teens_cased_fst[k] | (pynutil.delete("0") + digits_bare_cased_fst[k])
        )
        if k != "nom_sg":
            two_digit_cased_fsts_sfx[k] = two_digit_cased_fsts[k] + pynutil.delete(endings_cased[k])
            if two_digits_fst is None:
                two_digits_fst = two_digit_cased_fsts_sfx[k]
            else:
                two_digits_fst |= two_digit_cased_fsts_sfx[k]

    # bare hundreds
    bare_hundreds_fst = {}
    for k in digits_cased_fst:
        bare_hundred = pynini.cross("00", cuodi_cased[k])
        if k in ["ill_sg", "loc_sg"]:
            prefix_digit = (NEMO_DIGIT - "1") @ digits_cased_fst["gen_sg"]
            prefix_digit |= pynutil.delete("1")
        else:
            prefix_digit = (NEMO_DIGIT - "1") @ digits_cased_fst[k]
            prefix_digit |= pynutil.delete("1")
        if not deterministic:
            if k == "com_pl":
                prefix_digit |= (NEMO_DIGIT - "1") @ digits_cased_fst["gen_pl"]
            elif k == "ess":
                bare_hundred |= pynini.cross("00", "čuohtin")
            elif k == "nom_sg":
                bare_hundred |= pynini.cross("00", "čuohti")
        bare_hundreds_fst[k] = prefix_digit + spacer + bare_hundred

    def select_tens(tens_cased):
        return ((NEMO_DIGIT - "0") + pynini.accep("0")) @ tens_cased

    # 3 digit
    prefix_hundreds = digits_nom_prefix + pynutil.insert("čuođi")
    just_tens_nom = ((NEMO_DIGIT - "1" - "0") + pynutil.insert("0")) @ tens_cased_fst['nom_sg']
    hundreds_fst = {}
    for k in digits_cased_fst:
        hundreds_fst[k] = prefix_hundreds + pynutil.delete("0") + spacer + digits_bare_cased_fst[k]
        hundreds_fst[k] |= prefix_hundreds + spacer + teens_cased_fst[k]
        if k in ["loc_pl", "com_sg"]:
            hundreds_fst[k] |= prefix_hundreds + spacer + tens_cased_fst[k]
            if not deterministic:
                hundreds_fst[k] |= prefix_hundreds + spacer + just_tens_nom + spacer + digits_bare_cased_fst[k]
                hundreds_fst[k] |= prefix_hundreds + spacer + select_tens(tens_cased_fst[k])
        else:
            hundreds_fst[k] |= prefix_hundreds + spacer + just_tens_nom + spacer + digits_bare_cased_fst[k]
            hundreds_fst[k] |= prefix_hundreds + spacer + select_tens(tens_cased_fst[k])
            if not deterministic:
                hundreds_fst[k] |= prefix_hundreds + spacer + tens_cased_fst[k]

    # thousands
    bare_thousands = {}
    thousands = {}

    return {
        "tens": tens_cased_fst,
        "teens": teens_cased_fst,
        "digits": digits_bare_cased_fst,
        "zero": nolla_cased_fst,
        "two_digit_cased_fsts": two_digit_cased_fsts,
        "two_digit_cased_fsts_sfx": two_digit_cased_fsts_sfx,
        "two_digit_fst": two_digits_fst,
        "bare_hundreds": bare_hundreds_fst,
        "hundreds": hundreds_fst,
    }


def make_million(
    initial: str, at_least_one_non_zero_digit_no_one: 'pynini.FstLike', deterministic=True
) -> 'pynini.FstLike':
    spacer = make_spacer(deterministic)

    graph_million = pynutil.add_weight(pynini.cross("001", f"{initial}iljovdna") + spacer, -0.001)
    graph_million |= at_least_one_non_zero_digit_no_one + spacer + pynutil.insert(f"{initial}iljovnna") + spacer
    if not deterministic:
        graph_million |= pynutil.add_weight(pynini.cross("001", f"{initial}iljon") + spacer, -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", f"{initial}iljun") + spacer, -0.001)
        graph_million |= pynutil.add_weight(pynini.cross("001", f"{initial}illiuvdna") + spacer, -0.001)
        graph_million |= at_least_one_non_zero_digit_no_one + spacer + pynutil.insert(f"{initial}illiuvnna") + spacer
    graph_million |= pynutil.delete("000")
    return graph_million


def make_milliard(
    initial: str, at_least_one_non_zero_digit_no_one: 'pynini.FstLike', deterministic=True
) -> 'pynini.FstLike':
    spacer = make_spacer(deterministic)

    graph_milliard = pynutil.add_weight(pynini.cross("001", f"{initial}iljárda"), -0.001) + spacer
    graph_milliard |= at_least_one_non_zero_digit_no_one + spacer + pynutil.insert(f"{initial}iljárdda") + spacer
    graph_milliard |= pynutil.delete("000")
    return graph_milliard


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
            hundreds |= pynini.cross("1", "aktačuođi")

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

        graph_million = make_million("m", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_milliard = make_milliard("m", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_billion = make_million("b", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_billiard = make_milliard("b", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_trillion = make_million("tr", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_trilliard = make_milliard(
            "tr", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic
        )

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
