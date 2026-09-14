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
        mag = dict(load_labels(get_abs_path("data/numbers/magnitudes.tsv")))

        te_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        one_digit = pynini.union("1", "౧")
        del_one = pynutil.delete(one_digit)
        one_as_oka = (one_digit @ ties_one_suffix).optimize()
        digit_nx01 = (pynini.difference(NEMO_ALL_DIGIT, NEMO_ALL_ZERO | one_digit) @ digit).optimize()

        def union(*parts):
            return pynini.union(*parts).optimize()

        def exact_n(n, graph=exact_power):
            return pynini.compose(NEMO_ALL_DIGIT**n, graph).optimize()

        def mag_ins(key, space=True):
            return pynutil.insert((" " if space else "") + mag[key])

        ins_thou = mag_ins("thousand", False)
        ins_thous, ins_thous_pl = mag_ins("thousands_before"), mag_ins("thousands_plural")
        ins_lakh = mag_ins("lakh", False)
        ins_lakhs, ins_lakhs_pl = mag_ins("lakhs_before"), mag_ins("lakhs_plural")
        ins_koti, ins_koti_sp = mag_ins("crore", False), mag_ins("crore")
        ins_kotlu, ins_kotlu_pl = mag_ins("crores_before"), mag_ins("crores_plural")
        ins_hund, ins_hund_pl = mag_ins("hundreds_before"), mag_ins("hundreds_plural")

        hprefix_2d = union(
            pynini.compose(NEMO_DIGIT + NEMO_DIGIT, hundred_prefix),
            pynini.compose(te_digit + te_digit, hundred_prefix),
        )
        hprefix_1d = union(pynini.compose(NEMO_DIGIT, hundred_prefix), pynini.compose(te_digit, hundred_prefix))

        def make_teens(d_cls, zero_ch, dig_map):
            t = pynini.compose(d_cls + d_cls, teens)
            ti = (d_cls @ ties).optimize()
            return t | (ti + pynutil.delete(zero_ch)) | (ti + insert_space + dig_map)

        dig_ascii, dig_te = (NEMO_DIGIT @ digit).optimize(), (te_digit @ digit).optimize()
        dig_nx01_en = (pynini.difference(NEMO_DIGIT, pynini.union("0", "1")) @ digit).optimize()
        dig_nx01_te = (pynini.difference(te_digit, pynini.union("౦", "౧")) @ digit).optimize()
        teens_ties = union(make_teens(NEMO_DIGIT, "0", dig_ascii), make_teens(te_digit, "౦", dig_te))
        teens_x1 = union(make_teens(NEMO_DIGIT, "0", dig_nx01_en), make_teens(te_digit, "౦", dig_nx01_te))
        teens_oka = union(
            (NEMO_DIGIT @ ties) + insert_space + (NEMO_DIGIT @ ties_one_suffix),
            (te_digit @ ties) + insert_space + (te_digit @ ties_one_suffix),
        )
        teens_before = union(teens_oka, teens_x1)
        self.single_digits_graph = (digit | zero) + pynini.closure(insert_space + (digit | zero))
        delete_zero = pynutil.delete(NEMO_ALL_ZERO)
        zdel = {0: pynini.accep("")}

        for n in range(1, 8):
            zdel[n] = (zdel[n - 1] + delete_zero).optimize()

        def with_unit(prefix, suf, zeros):
            return prefix + suf if zeros == 0 else prefix + zdel[zeros] + suf

        def with_rem(prefix, suf, zeros, sub):
            return prefix + (suf if zeros == 0 else suf + zdel[zeros]) + insert_space + sub

        def mag_forms(prefix, suf, ladder, head=None, head_z=None):
            g = with_unit(prefix, head, head_z) if head is not None else None
            for zeros, sub in ladder:
                g = with_rem(prefix, suf, zeros, sub) if g is None else g | with_rem(prefix, suf, zeros, sub)
            return g

        def prio(a, b):
            return plurals._priority_union(a, b, NEMO_SIGMA)

        def ties_scale(oka_suf, other_suf, ladder, head_oka, head_other, head_z):
            return union(
                mag_forms(teens_oka, oka_suf, ladder, head_oka, head_z),
                mag_forms(teens_x1, other_suf, ladder, head_other, head_z),
            )

        def scale(exact, sg, before, head, spaced, zeros, ladder, one_ladder=None, extra=None, ten_oka=None):
            one_ladder = ladder if one_ladder is None else one_ladder
            g = exact | mag_forms(del_one, sg, one_ladder)
            if extra is not None:
                g = g | extra
            g = union(g, mag_forms(digit_nx01, before, ladder, head, zeros))
            ten = ties_scale(ten_oka or before, before, ladder, spaced, head, zeros)
            return g, ten

        def oka_mult(sg, before, remainders):
            def side(use_digit):
                parts = []
                for zeros, rem in remainders:
                    body = zdel[zeros] + insert_space + rem
                    if use_digit:
                        parts += [del_one + sg + body, digit_nx01 + before + body]
                    else:
                        parts += [teens_oka + before + body, teens_x1 + before + body]
                return union(*parts)

            return prio(side(True), side(False)).optimize()

        def crore_unit(oka, other, other_head=None, nested=False):
            if nested:
                oka_suf, oka_head = ins_koti_sp, ins_koti_sp
            else:
                oka_suf, oka_head = ins_kotlu, ins_kotlu_pl
            return prio(
                mag_forms(oka, oka_suf, crore_ladder, oka_head, 7),
                mag_forms(other, ins_kotlu, crore_ladder, other_head or ins_kotlu_pl, 7),
            ).optimize()

        def outer_crores(count):
            return mag_forms(count, ins_kotlu, crore_ladder, ins_kotlu_pl, 7).optimize()

        graph_hundreds = union(
            exact_n(3),
            hprefix_2d + digit,
            hprefix_1d + teens_ties,
            with_unit(digit_nx01, ins_hund_pl, 2),
            with_rem(digit_nx01, ins_hund, 1, digit),
            with_rem(digit_nx01, ins_hund, 0, teens_ties),
        )
        thousand_ladder = [(2, digit), (1, teens_ties), (0, graph_hundreds)]
        graph_thousands, graph_ten_thousands = scale(
            exact_n(4), ins_thou, ins_thous, ins_thous_pl, ins_thous_pl, 3, thousand_ladder
        )
        lakh_ladder = [
            (4, digit),
            (3, teens_ties),
            (2, graph_hundreds),
            (1, graph_thousands),
            (0, graph_ten_thousands),
        ]
        graph_lakhs, graph_ten_lakhs = scale(
            exact_n(6),
            ins_lakh,
            ins_lakhs,
            ins_lakhs_pl,
            ins_lakhs_pl,
            5,
            lakh_ladder,
            one_ladder=lakh_ladder[1:],
            extra=with_rem(del_one, ins_lakh, 4, digit),
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
        graph_crores, graph_ten_crores = scale(
            exact_n(8), ins_koti, ins_kotlu, ins_kotlu_pl, ins_kotlu_pl, 7, crore_ladder
        )
        hundred_one = union(
            hprefix_1d + pynutil.delete(NEMO_ALL_ZERO) + one_as_oka,
            digit_nx01 + pynutil.delete(NEMO_ALL_ZERO) + hundreds_before_one,
        )
        hundred_crore = union(
            exact_n(3),
            hundred_one,
            hprefix_2d + one_as_oka,
            hprefix_2d + digit_nx01,
            hprefix_1d + teens_before,
            with_unit(digit_nx01, ins_hund, 2),
            with_rem(digit_nx01, ins_hund, 1, digit_nx01),
            with_rem(digit_nx01, ins_hund, 0, teens_before),
        )
        thou_crore_ladder = [(2, digit_nx01), (1, teens_before), (0, hundred_crore)]
        thousand_one = oka_mult(ins_thou, ins_thous, [(2, one_as_oka), (0, hundred_one)])
        thousand_crore, ten_thousand_crore = scale(
            exact_n(4), ins_thou, ins_thous, ins_thous, ins_thous, 3, thou_crore_ladder
        )
        graph_hundred_crores = union(
            crore_unit(hundred_one, hundred_crore), with_rem(hundred_crore, ins_kotlu, 0, graph_crores)
        )
        graph_thousand_crores = crore_unit(thousand_one, union(thousand_crore, ten_thousand_crore))
        graph_ten_thousand_crores = crore_unit(thousand_one, ten_thousand_crore)
        lakh_crore_ladder = [
            (4, digit_nx01),
            (3, teens_before),
            (2, hundred_crore),
            (1, thousand_crore),
            (0, ten_thousand_crore),
        ]
        lakh_one = oka_mult(
            ins_lakh,
            ins_lakhs,
            [(4, one_as_oka), (2, hundred_one), (1, exact_n(4, thousand_one)), (0, exact_n(5, thousand_one))],
        )
        lakh_crore, ten_lakh_crore = scale(exact_n(6), ins_lakh, ins_lakhs, ins_lakhs, ins_lakhs, 5, lakh_crore_ladder)
        graph_lakh_crores = crore_unit(lakh_one, union(lakh_crore, ten_lakh_crore))
        ten_lakh_rem = ties_scale(
            ins_lakhs,
            ins_lakhs,
            [
                (4, digit),
                (3, teens_before),
                (2, hundred_crore),
                (1, thousand_crore),
                (0, ten_thousand_crore),
            ],
            ins_lakhs,
            ins_lakhs,
            5,
        )
        koti_ladder = [
            (6, digit),
            (5, teens_before),
            (4, hundred_crore),
            (3, thousand_crore),
            (2, ten_thousand_crore),
            (1, lakh_crore),
            (0, ten_lakh_rem),
        ]
        crore_one = oka_mult(
            ins_koti,
            ins_kotlu,
            [(6, one_as_oka), (4, hundred_one), (1, exact_n(6, lakh_one)), (0, exact_n(7, lakh_one))],
        )
        except_one_ladder = [(2, digit_nx01), (1, teens_x1)]
        hundred_amt = union(
            exact_n(3),
            hprefix_2d + digit_nx01,
            hprefix_1d + teens_x1,
            with_unit(digit_nx01, ins_hund, 2),
            with_rem(digit_nx01, ins_hund, 1, digit_nx01),
        )
        thousand_amt = union(*scale(exact_n(4), ins_thou, ins_thous, ins_thous, ins_thous, 3, except_one_ladder))
        ten_thousand_amt = ties_scale(
            ins_thous, ins_thous, except_one_ladder + [(0, hundred_amt)], ins_thous, ins_thous, 3
        )
        crore_one_10 = with_rem(hundred_amt, ins_kotlu, 6, one_as_oka).optimize()
        crore_one_11 = with_rem(thousand_amt, ins_kotlu, 6, one_as_oka).optimize()
        crore_one_12 = with_rem(ten_thousand_amt, ins_kotlu, 6, one_as_oka).optimize()
        crore_one_10 = union(crore_one_10, with_rem(hundred_crore, ins_kotlu, 0, exact_n(7, lakh_one))).optimize()
        crore_one_11 = union(crore_one_11, with_rem(thousand_crore, ins_kotlu, 0, exact_n(7, lakh_one))).optimize()
        crore_one_12 = union(crore_one_12, with_rem(ten_thousand_crore, ins_kotlu, 0, exact_n(7, lakh_one))).optimize()

        crore_before, ten_crore_before = scale(exact_n(8), ins_koti, ins_kotlu, ins_kotlu, ins_kotlu, 7, koti_ladder)
        ten_lakh_crore_count = prio(
            union(
                with_rem(del_one, ins_koti, 0, ten_lakh_rem),
                with_rem(digit_nx01, ins_kotlu, 0, ten_lakh_rem),
                with_rem(teens_oka, ins_kotlu, 0, ten_lakh_rem),
                with_rem(teens_x1, ins_kotlu, 0, ten_lakh_rem),
            ),
            union(crore_before, ten_crore_before),
        ).optimize()
        graph_ten_lakh_crores = crore_unit(exact_n(8, crore_one), ten_lakh_crore_count, nested=True)
        other_nine = mag_forms(teens_x1, ins_kotlu, koti_ladder, ins_kotlu, 7)

        graph_crore_crores = prio(
            union(
                mag_forms(exact_n(9, crore_one), ins_koti_sp, crore_ladder, ins_koti_sp, 7),
                outer_crores(with_unit(teens_oka, ins_kotlu, 7)),
            ),
            outer_crores(
                union(
                    mag_forms(teens_oka, ins_kotlu, koti_ladder, ins_kotlu_pl, 7),
                    other_nine,
                )
            ),
        ).optimize()
        graph_ten_crore_crores = prio(
            union(
                mag_forms(crore_one_10, ins_koti_sp, crore_ladder, ins_koti_sp, 7),
                outer_crores(with_unit(hundred_one, ins_kotlu, 7)),
            ),
            outer_crores(
                union(
                    mag_forms(hundred_one, ins_kotlu, koti_ladder, ins_kotlu_pl, 7),
                    mag_forms(hundred_crore, ins_kotlu, koti_ladder, ins_kotlu, 7),
                )
            ),
        ).optimize()
        hcc_oka = mag_forms(teens_oka, ins_thous, thou_crore_ladder, ins_thous, 3)
        hcc_x1 = union(thousand_crore, mag_forms(teens_x1, ins_thous, thou_crore_ladder, ins_thous, 3))
        graph_hundred_crore_crores = prio(
            union(
                mag_forms(crore_one_12, ins_koti_sp, crore_ladder, ins_koti_sp, 7),
                mag_forms(crore_one_11, ins_koti_sp, crore_ladder, ins_koti_sp, 7),
                mag_forms(crore_one_10 + zdel[1], ins_koti_sp, crore_ladder, ins_koti_sp, 7),
                outer_crores(with_unit(thousand_one, ins_kotlu, 7)),
                outer_crores(with_unit(hcc_oka, ins_kotlu, 7)),
            ),
            outer_crores(
                union(
                    mag_forms(thousand_one, ins_kotlu, koti_ladder, ins_kotlu_pl, 7),
                    mag_forms(hcc_oka, ins_kotlu, koti_ladder, ins_kotlu_pl, 7),
                    mag_forms(hcc_x1, ins_kotlu, koti_ladder, ins_kotlu, 7),
                )
            ),
        ).optimize()

        graph = union(
            digit,
            zero,
            teens_ties,
            graph_hundreds,
            graph_thousands,
            graph_ten_thousands,
            graph_lakhs,
            graph_ten_lakhs,
            graph_crores,
            graph_ten_crores,
            exact_n(10, graph_hundred_crores),
            exact_n(11, graph_thousand_crores),
            exact_n(12, graph_ten_thousand_crores),
            exact_n(13, graph_lakh_crores),
            exact_n(14, graph_lakh_crores),
            exact_n(15, graph_ten_lakh_crores),
            exact_n(16, graph_crore_crores),
            exact_n(17, graph_ten_crore_crores),
            exact_n(18, graph_hundred_crore_crores),
            exact_n(19, graph_hundred_crore_crores),
        )
        graph = pynini.compose(pynini.closure(NEMO_DIGIT, 1) | pynini.closure(te_digit, 1), graph)

        leading_zeros = pynini.compose(
            (pynini.closure("0", 1) + pynini.closure(NEMO_DIGIT))
            | (pynini.closure("౦", 1) + pynini.closure(te_digit)),
            self.single_digits_graph,
        )
        sep, two, three = pynutil.delete(","), NEMO_ALL_DIGIT**2, NEMO_ALL_DIGIT**3
        grouped = pynini.compose(
            union(
                pynini.closure(NEMO_ALL_DIGIT, 1, 2) + pynini.closure(sep + two) + sep + three,
                pynini.closure(NEMO_ALL_DIGIT, 1, 3) + pynini.closure(sep + three, 1),
            ),
            graph,
        ).optimize()
        final = union(graph, leading_zeros, grouped)
        minus = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)
        self.final_graph = final.optimize()
        self.fst = self.add_tokens(minus + pynutil.insert('integer: "') + self.final_graph + pynutil.insert('"'))
