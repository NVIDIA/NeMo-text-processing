# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_ALL_ZERO,
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path, load_labels


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -౨౩ -> cardinal { negative: "true"  integer: "ఇరవై మూడు" }

    Covers numbers up to 19 digits by composing crore (కోటి) groups
    (through hundred crore crores (వంద కోట్ల కోట్లు) / 10^17).

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        exact_power = pynini.string_file(get_abs_path("data/numbers/exact_power.tsv"))
        hundred_prefix = pynini.string_file(get_abs_path("data/numbers/hundred_prefix.tsv"))
        ties_one_suffix = pynini.string_file(get_abs_path("data/numbers/ties_one_suffix.tsv"))
        hundreds_before_one = pynini.string_file(get_abs_path("data/numbers/hundreds_before_one.tsv"))
        magnitude = {k: v for k, v in load_labels(get_abs_path("data/numbers/magnitudes.tsv"))}

        te_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        exact_hundred = pynini.compose(NEMO_ALL_DIGIT**3, exact_power).optimize()
        exact_thousand = pynini.compose(NEMO_ALL_DIGIT**4, exact_power).optimize()
        exact_lakh = pynini.compose(NEMO_ALL_DIGIT**6, exact_power).optimize()
        exact_crore = pynini.compose(NEMO_ALL_DIGIT**8, exact_power).optimize()
        hundred_prefix_ten = (
            pynini.compose(NEMO_DIGIT + NEMO_DIGIT, hundred_prefix)
            | pynini.compose(te_digit + te_digit, hundred_prefix)
        ).optimize()
        hundred_prefix_one = (
            pynini.compose(NEMO_DIGIT, hundred_prefix) | pynini.compose(te_digit, hundred_prefix)
        ).optimize()

        ins_hundreds_plural = pynutil.insert(" " + magnitude["hundreds_plural"])
        ins_hundreds_before = pynutil.insert(" " + magnitude["hundreds_before"])
        ins_thousand = pynutil.insert(magnitude["thousand"])
        ins_thousand_spaced = pynutil.insert(" " + magnitude["thousand"])
        ins_thousands_plural = pynutil.insert(" " + magnitude["thousands_plural"])
        ins_thousands_before = pynutil.insert(" " + magnitude["thousands_before"])
        ins_lakh = pynutil.insert(magnitude["lakh"])
        ins_lakh_spaced = pynutil.insert(" " + magnitude["lakh"])
        ins_lakha_digit = pynutil.insert(magnitude["lakh_before_digit"])
        ins_lakhs_plural = pynutil.insert(" " + magnitude["lakhs_plural"])
        ins_lakhs_before = pynutil.insert(" " + magnitude["lakhs_before"])
        ins_crore = pynutil.insert(magnitude["crore"])
        ins_crore_spaced = pynutil.insert(" " + magnitude["crore"])
        ins_crores_plural = pynutil.insert(" " + magnitude["crores_plural"])
        ins_crores_before = pynutil.insert(" " + magnitude["crores_before"])

        teens_en = pynini.compose(NEMO_DIGIT + NEMO_DIGIT, teens).optimize()
        teens_te = pynini.compose(te_digit + te_digit, teens).optimize()
        digit_en = (NEMO_DIGIT @ digit).optimize()
        digit_te = (te_digit @ digit).optimize()
        ties_en = (NEMO_DIGIT @ ties).optimize()
        ties_te = (te_digit @ ties).optimize()

        one_digit = pynini.union("1", "౧")
        digit_except_one = (pynini.difference(NEMO_ALL_DIGIT, NEMO_ALL_ZERO | one_digit) @ digit).optimize()
        digit_except_one_en = (pynini.difference(NEMO_DIGIT, pynini.union("0", "1")) @ digit).optimize()
        digit_except_one_te = (pynini.difference(te_digit, pynini.union("౦", "౧")) @ digit).optimize()
        one_as_oka = (one_digit @ ties_one_suffix).optimize()
        one_prefix = pynutil.delete(one_digit)

        teens_ties_en = teens_en | (ties_en + pynutil.delete("0")) | (ties_en + insert_space + digit_en)
        teens_ties_te = teens_te | (ties_te + pynutil.delete("౦")) | (ties_te + insert_space + digit_te)
        teens_ties = pynini.union(teens_ties_te, teens_ties_en)
        ties_one_suffix_en = (NEMO_DIGIT @ ties_one_suffix).optimize()
        ties_one_suffix_te = (te_digit @ ties_one_suffix).optimize()
        teens_ties_thousand = (
            (ties_en + insert_space + ties_one_suffix_en) | (ties_te + insert_space + ties_one_suffix_te)
        ).optimize()
        teens_ties_except_one = pynini.union(
            teens_en | (ties_en + pynutil.delete("0")) | (ties_en + insert_space + digit_except_one_en),
            teens_te | (ties_te + pynutil.delete("౦")) | (ties_te + insert_space + digit_except_one_te),
        ).optimize()

        single_digit_graph = digit | zero
        self.single_digits_graph = single_digit_graph + pynini.closure(insert_space + single_digit_graph)

        delete_zero = pynutil.delete(NEMO_ALL_ZERO)
        zero_pow = {0: pynini.accep("")}
        for _n in range(1, 8):
            zero_pow[_n] = (zero_pow[_n - 1] + delete_zero).optimize()

        def create_graph_suffix(prefix, suffix, zeros_counts):
            return prefix + suffix if zeros_counts == 0 else prefix + zero_pow[zeros_counts] + suffix

        def create_larger_number_graph(prefix, suffix, zeros_counts, sub_graph):
            mid = suffix if zeros_counts == 0 else suffix + zero_pow[zeros_counts]
            return prefix + mid + insert_space + sub_graph

        def build_group(prefix, rung_suffix, ladder, head_suffix=None, head_zeros=None):
            graph = create_graph_suffix(prefix, head_suffix, head_zeros) if head_suffix is not None else None
            for zeros, sub in ladder:
                rung = create_larger_number_graph(prefix, rung_suffix, zeros, sub)
                graph = rung if graph is None else graph | rung
            return graph

        def prefer(primary, secondary):
            return plurals._priority_union(primary, secondary, NEMO_SIGMA)

        def tie_pair(oka_suffix, other_suffix, ladder, head_oka, head_other, head_zeros):
            return (
                build_group(teens_ties_thousand, oka_suffix, ladder, head_suffix=head_oka, head_zeros=head_zeros)
                | build_group(
                    teens_ties_except_one, other_suffix, ladder, head_suffix=head_other, head_zeros=head_zeros
                )
            ).optimize()

        def oka_count_prefix(singular_ins, before_ins, remainders):
            """Build …01 count prefixes: singular/before magnitude + (zeros, remainder→ఒక)."""
            pieces = []
            for zeros, rem in remainders:
                body = zero_pow[zeros] + insert_space + rem
                digit_pref = pynini.union(
                    one_prefix + singular_ins + body,
                    digit_except_one + before_ins + body,
                )
                ties_pref = pynini.union(
                    teens_ties_thousand + before_ins + body,
                    teens_ties_except_one + before_ins + body,
                )
                pieces.append(prefer(digit_pref, ties_pref))
            return pynini.union(*pieces).optimize()

        def crore_graph(oka_prefix, other_prefix, ladder=None, other_head=None):
            ladder = crore_ladder if ladder is None else ladder
            other_head = ins_crores_plural if other_head is None else other_head
            oka = build_group(oka_prefix, ins_crore_spaced, ladder, head_suffix=ins_crore_spaced, head_zeros=7)
            other = build_group(other_prefix, ins_crores_before, ladder, head_suffix=other_head, head_zeros=7)
            return prefer(oka, other).optimize()

        graph_hundreds = (
            exact_hundred
            | hundred_prefix_ten + digit
            | hundred_prefix_one + teens_ties
            | create_graph_suffix(digit_except_one, ins_hundreds_plural, 2)
            | create_larger_number_graph(digit_except_one, ins_hundreds_before, 1, digit)
            | create_larger_number_graph(digit_except_one, ins_hundreds_before, 0, teens_ties)
        ).optimize()

        thousand_ladder = [(2, digit), (1, teens_ties), (0, graph_hundreds)]
        graph_thousands = (
            exact_thousand
            | build_group(one_prefix, ins_thousand, thousand_ladder)
            | build_group(
                digit_except_one, ins_thousands_before, thousand_ladder, head_suffix=ins_thousands_plural, head_zeros=3
            )
        ).optimize()
        graph_ten_thousands = tie_pair(
            ins_thousands_before,
            ins_thousands_before,
            thousand_ladder,
            ins_thousand_spaced,
            ins_thousands_plural,
            3,
        )

        lakh_ladder = [
            (4, digit),
            (3, teens_ties),
            (2, graph_hundreds),
            (1, graph_thousands),
            (0, graph_ten_thousands),
        ]
        graph_lakhs = (
            exact_lakh
            | create_larger_number_graph(one_prefix, ins_lakha_digit, 4, digit)
            | build_group(one_prefix, ins_lakh, lakh_ladder[1:])
            | build_group(digit_except_one, ins_lakhs_before, lakh_ladder, head_suffix=ins_lakhs_plural, head_zeros=5)
        ).optimize()
        graph_ten_lakhs = tie_pair(
            ins_lakhs_before, ins_lakhs_before, lakh_ladder, ins_lakh_spaced, ins_lakhs_plural, 5
        )

        crore_ladder = [
            (6, digit),
            (5, teens_ties),
            (4, graph_hundreds),
            (3, graph_thousands),
            (2, graph_ten_thousands),
            (1, graph_lakhs),
            (0, graph_ten_lakhs),
        ]
        graph_crores = (
            exact_crore
            | build_group(one_prefix, ins_crore, crore_ladder)
            | build_group(
                digit_except_one, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            )
        ).optimize()
        graph_ten_crores = tie_pair(
            ins_crore_spaced, ins_crores_before, crore_ladder, ins_crore_spaced, ins_crores_plural, 7
        )

        hundred_crore_prefix = (
            exact_hundred
            | (hundred_prefix_ten + digit_except_one)
            | (hundred_prefix_one + teens_ties)
            | create_graph_suffix(digit_except_one, ins_hundreds_before, 2)
            | create_larger_number_graph(digit_except_one, ins_hundreds_before, 1, digit_except_one)
            | create_larger_number_graph(digit_except_one, ins_hundreds_before, 0, teens_ties)
        ).optimize()
        hundred_one_crore_prefix = (
            hundred_prefix_one + pynutil.delete(NEMO_ALL_ZERO) + one_as_oka
            | digit_except_one + pynutil.delete(NEMO_ALL_ZERO) + hundreds_before_one
        ).optimize()

        thousand_crore_ladder = [(2, digit_except_one), (1, teens_ties), (0, hundred_crore_prefix)]
        thousand_one_crore_prefix = oka_count_prefix(
            ins_thousand, ins_thousands_before, [(2, one_as_oka), (0, hundred_one_crore_prefix)]
        )
        thousand_crore_prefix = (
            exact_thousand
            | build_group(
                digit_except_one,
                ins_thousands_before,
                thousand_crore_ladder,
                head_suffix=ins_thousands_before,
                head_zeros=3,
            )
            | build_group(one_prefix, ins_thousand, thousand_crore_ladder)
        ).optimize()
        ten_thousand_crore_prefix = tie_pair(
            ins_thousands_before,
            ins_thousands_before,
            thousand_crore_ladder,
            ins_thousand_spaced,
            ins_thousands_before,
            3,
        )
        crore_count_prefix = (thousand_crore_prefix | ten_thousand_crore_prefix).optimize()

        graph_hundred_crores = (
            build_group(
                hundred_one_crore_prefix, ins_crore_spaced, crore_ladder, head_suffix=ins_crore_spaced, head_zeros=7
            )
            | build_group(
                hundred_crore_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            )
            | create_larger_number_graph(hundred_crore_prefix, ins_crores_before, 0, graph_crores)
        ).optimize()
        graph_thousand_crores = crore_graph(thousand_one_crore_prefix, crore_count_prefix)
        graph_ten_thousand_crores = crore_graph(thousand_one_crore_prefix, ten_thousand_crore_prefix)

        lakh_crore_ladder = [
            (4, digit_except_one),
            (3, teens_ties),
            (2, hundred_crore_prefix),
            (1, graph_thousands),
            (0, ten_thousand_crore_prefix),
        ]
        lakh_one_crore_prefix = oka_count_prefix(
            ins_lakh,
            ins_lakhs_before,
            [(5, one_as_oka), (2, hundred_one_crore_prefix), (0, thousand_one_crore_prefix)],
        )
        lakh_crore_prefix = (
            exact_lakh
            | build_group(one_prefix, ins_lakh, lakh_crore_ladder)
            | build_group(
                digit_except_one, ins_lakhs_before, lakh_crore_ladder, head_suffix=ins_lakhs_before, head_zeros=5
            )
        ).optimize()
        ten_lakh_crore_prefix = tie_pair(
            ins_lakhs_before, ins_lakhs_before, lakh_crore_ladder, ins_lakh_spaced, ins_lakhs_before, 5
        )
        graph_lakh_crores = crore_graph(lakh_one_crore_prefix, lakh_crore_prefix | ten_lakh_crore_prefix)

        ten_lakh_crore_lakh_remainder = tie_pair(
            ins_lakhs_before, ins_lakhs_before, lakh_ladder, ins_lakhs_before, ins_lakhs_before, 5
        )
        koti_ladder = [
            (6, digit),
            (5, teens_ties),
            (4, graph_hundreds),
            (3, graph_thousands),
            (2, graph_ten_thousands),
            (1, graph_lakhs),
            (0, ten_lakh_crore_lakh_remainder),
        ]

        ten_lakh_crore_count_prefix = prefer(
            create_larger_number_graph(one_prefix, ins_crore, 0, ten_lakh_crore_lakh_remainder)
            | create_larger_number_graph(digit_except_one, ins_crores_before, 0, ten_lakh_crore_lakh_remainder)
            | create_larger_number_graph(teens_ties_thousand, ins_crore_spaced, 0, ten_lakh_crore_lakh_remainder)
            | create_larger_number_graph(teens_ties_except_one, ins_crores_before, 0, ten_lakh_crore_lakh_remainder),
            graph_crores | graph_ten_crores,
        ).optimize()
        graph_ten_lakh_crores = build_group(
            ten_lakh_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
        ).optimize()

        crore_crore_count_prefix = tie_pair(
            ins_crore_spaced, ins_crores_before, koti_ladder, ins_crore_spaced, ins_crores_before, 7
        )
        graph_crore_crores = build_group(
            crore_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
        ).optimize()

        ten_crore_crore_count_prefix = prefer(
            create_graph_suffix(hundred_one_crore_prefix, ins_crore_spaced, 7)
            | build_group(
                hundred_one_crore_prefix, ins_crore_spaced, koti_ladder, head_suffix=ins_crore_spaced, head_zeros=7
            ),
            build_group(
                hundred_crore_prefix, ins_crores_before, koti_ladder, head_suffix=ins_crores_before, head_zeros=7
            ),
        ).optimize()
        graph_ten_crore_crores = build_group(
            ten_crore_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
        ).optimize()

        hundred_crore_crore_oka_count_prefix = build_group(
            teens_ties_thousand,
            ins_thousands_before,
            thousand_crore_ladder,
            head_suffix=ins_thousand_spaced,
            head_zeros=3,
        ).optimize()
        hundred_crore_crore_other_count_prefix = (
            thousand_crore_prefix
            | build_group(
                teens_ties_except_one,
                ins_thousands_before,
                thousand_crore_ladder,
                head_suffix=ins_thousands_before,
                head_zeros=3,
            )
        ).optimize()
        hundred_crore_crore_count_prefix = prefer(
            build_group(
                thousand_one_crore_prefix, ins_crore_spaced, koti_ladder, head_suffix=ins_crore_spaced, head_zeros=7
            )
            | build_group(
                hundred_crore_crore_oka_count_prefix,
                ins_crore_spaced,
                koti_ladder,
                head_suffix=ins_crore_spaced,
                head_zeros=7,
            ),
            build_group(
                hundred_crore_crore_other_count_prefix,
                ins_crores_before,
                koti_ladder,
                head_suffix=ins_crores_before,
                head_zeros=7,
            ),
        ).optimize()
        graph_hundred_crore_crores = build_group(
            hundred_crore_crore_count_prefix,
            ins_crores_before,
            crore_ladder,
            head_suffix=ins_crores_plural,
            head_zeros=7,
        ).optimize()

        def exact_digits(n, graph):
            return pynini.compose(NEMO_ALL_DIGIT**n, graph)

        graph_without_leading_zeros = (
            digit
            | zero
            | teens_ties
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | exact_digits(10, graph_hundred_crores)
            | exact_digits(11, graph_thousand_crores)
            | exact_digits(12, graph_ten_thousand_crores)
            | exact_digits(13, graph_lakh_crores)
            | exact_digits(14, graph_lakh_crores)
            | exact_digits(15, graph_ten_lakh_crores)
            | exact_digits(16, graph_crore_crores)
            | exact_digits(17, graph_ten_crore_crores)
            | exact_digits(18, graph_hundred_crore_crores)
            | exact_digits(19, graph_hundred_crore_crores)
        )
        same_script_number = pynini.closure(NEMO_DIGIT, 1) | pynini.closure(te_digit, 1)
        graph_without_leading_zeros = pynini.compose(same_script_number, graph_without_leading_zeros)

        cardinal_with_leading_zeros = pynini.compose(
            (pynini.closure("0", 1) + pynini.closure(NEMO_DIGIT))
            | (pynini.closure("౦", 1) + pynini.closure(te_digit)),
            self.single_digits_graph,
        )
        delete_separator = pynutil.delete(",")
        two_digits = NEMO_ALL_DIGIT + NEMO_ALL_DIGIT
        three_digits = NEMO_ALL_DIGIT + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT
        indian_grouping = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 2)
            + pynini.closure(delete_separator + two_digits)
            + delete_separator
            + three_digits
        )
        western_grouping = pynini.closure(NEMO_ALL_DIGIT, 1, 3) + pynini.closure(delete_separator + three_digits, 1)
        cardinal_with_separators = pynini.compose(
            (indian_grouping | western_grouping).optimize(), graph_without_leading_zeros
        ).optimize()

        final_graph = graph_without_leading_zeros | cardinal_with_leading_zeros | cardinal_with_separators
        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph)
