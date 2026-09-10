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
        one_prefix = pynutil.delete(one_digit)
        one_as_oka = (one_digit @ ties_one_suffix).optimize()
        digit_x1 = (pynini.difference(NEMO_ALL_DIGIT, NEMO_ALL_ZERO | one_digit) @ digit).optimize()

        def U(*parts):
            return pynini.union(*parts).optimize()

        def exact_n(n, graph=exact_power):
            return pynini.compose(NEMO_ALL_DIGIT**n, graph).optimize()

        def ins(key, space=True):
            return pynutil.insert((" " if space else "") + mag[key])

        i_thou, i_thou_sp = ins("thousand", False), ins("thousand")
        i_thous, i_thous_pl = ins("thousands_before"), ins("thousands_plural")
        i_lakh, i_lakh_sp = ins("lakh", False), ins("lakh")
        i_lakha, i_lakhs, i_lakhs_pl = (
            pynutil.insert(mag["lakh_before_digit"]),
            ins("lakhs_before"),
            ins("lakhs_plural"),
        )
        i_koti, i_koti_sp = ins("crore", False), ins("crore")
        i_kotlu, i_kotlu_pl = ins("crores_before"), ins("crores_plural")
        i_vandalu, i_vandalu_pl = ins("hundreds_before"), ins("hundreds_plural")

        hp_ten = U(
            pynini.compose(NEMO_DIGIT + NEMO_DIGIT, hundred_prefix),
            pynini.compose(te_digit + te_digit, hundred_prefix),
        )
        hp_one = U(pynini.compose(NEMO_DIGIT, hundred_prefix), pynini.compose(te_digit, hundred_prefix))

        def teens_ties_of(d_cls, zero_ch, dig_map):
            t = pynini.compose(d_cls + d_cls, teens)
            ti = (d_cls @ ties).optimize()
            return t | (ti + pynutil.delete(zero_ch)) | (ti + insert_space + dig_map)

        dig_en, dig_te = (NEMO_DIGIT @ digit).optimize(), (te_digit @ digit).optimize()
        dig_x1_en = (pynini.difference(NEMO_DIGIT, pynini.union("0", "1")) @ digit).optimize()
        dig_x1_te = (pynini.difference(te_digit, pynini.union("౦", "౧")) @ digit).optimize()
        teens_ties = U(teens_ties_of(NEMO_DIGIT, "0", dig_en), teens_ties_of(te_digit, "౦", dig_te))
        teens_ties_x1 = U(teens_ties_of(NEMO_DIGIT, "0", dig_x1_en), teens_ties_of(te_digit, "౦", dig_x1_te))
        teens_ties_oka = U(
            (NEMO_DIGIT @ ties) + insert_space + (NEMO_DIGIT @ ties_one_suffix),
            (te_digit @ ties) + insert_space + (te_digit @ ties_one_suffix),
        )

        self.single_digits_graph = (digit | zero) + pynini.closure(insert_space + (digit | zero))

        delete_zero = pynutil.delete(NEMO_ALL_ZERO)
        z = {0: pynini.accep("")}
        for n in range(1, 8):
            z[n] = (z[n - 1] + delete_zero).optimize()

        def suffix(prefix, suf, zeros):
            return prefix + suf if zeros == 0 else prefix + z[zeros] + suf

        def rung(prefix, suf, zeros, sub):
            return prefix + (suf if zeros == 0 else suf + z[zeros]) + insert_space + sub

        def group(prefix, suf, ladder, head=None, head_z=None):
            g = suffix(prefix, head, head_z) if head is not None else None
            for zeros, sub in ladder:
                g = rung(prefix, suf, zeros, sub) if g is None else g | rung(prefix, suf, zeros, sub)
            return g

        def prefer(a, b):
            return plurals._priority_union(a, b, NEMO_SIGMA)

        def ties_group(oka_suf, other_suf, ladder, head_oka, head_other, head_z):
            return U(
                group(teens_ties_oka, oka_suf, ladder, head_oka, head_z),
                group(teens_ties_x1, other_suf, ladder, head_other, head_z),
            )

        def band(exact, sg, before, head, spaced, zeros, ladder, one_ladder=None, extra=None, ten_oka=None):
            """Digit magnitude (+ optional teens ties). head is plural (standalone) or before (crore count)."""
            one_ladder = ladder if one_ladder is None else one_ladder
            g = exact | group(one_prefix, sg, one_ladder)
            if extra is not None:
                g = g | extra
            g = U(g, group(digit_x1, before, ladder, head, zeros))
            ten = ties_group(ten_oka or before, before, ladder, spaced, head, zeros)
            return g, ten

        def oka_count(sg, before, remainders):
            """…01 multipliers: 1+sg / N+before / ties+before + zeros + rem→ఒక."""

            def side(use_digit):
                parts = []
                for zeros, rem in remainders:
                    body = z[zeros] + insert_space + rem
                    if use_digit:
                        parts += [one_prefix + sg + body, digit_x1 + before + body]
                    else:
                        parts += [teens_ties_oka + before + body, teens_ties_x1 + before + body]
                return U(*parts)

            return prefer(side(True), side(False)).optimize()

        def crore_of(oka, other, other_head=None):
            """10–14: …01 → కోటి; else → కోట్లు."""
            return prefer(
                group(oka, i_koti_sp, crore_ladder, i_koti_sp, 7),
                group(other, i_kotlu, crore_ladder, other_head or i_kotlu_pl, 7),
            ).optimize()

        def kotlu(count):
            """15–19 outer unit: always కోట్లు."""
            return group(count, i_kotlu, crore_ladder, i_kotlu_pl, 7).optimize()

        def oka_koti(*prefs, bare=(), pad=()):
            """ఒక → ఒక కోటి (+ koti_ladder / bare / padded zeros)."""
            parts = [group(p, i_koti_sp, koti_ladder, i_koti_sp, 7) for p in prefs]
            parts += [p + i_koti_sp for p in bare]
            parts += [suffix(p, i_koti_sp, n) for p, n in pad]
            return U(*parts)

        graph_hundreds = U(
            exact_n(3),
            hp_ten + digit,
            hp_one + teens_ties,
            suffix(digit_x1, i_vandalu_pl, 2),
            rung(digit_x1, i_vandalu, 1, digit),
            rung(digit_x1, i_vandalu, 0, teens_ties),
        )
        thousand_ladder = [(2, digit), (1, teens_ties), (0, graph_hundreds)]
        graph_thousands, graph_ten_thousands = band(
            exact_n(4), i_thou, i_thous, i_thous_pl, i_thou_sp, 3, thousand_ladder
        )
        lakh_ladder = [
            (4, digit),
            (3, teens_ties),
            (2, graph_hundreds),
            (1, graph_thousands),
            (0, graph_ten_thousands),
        ]
        graph_lakhs, graph_ten_lakhs = band(
            exact_n(6),
            i_lakh,
            i_lakhs,
            i_lakhs_pl,
            i_lakh_sp,
            5,
            lakh_ladder,
            one_ladder=lakh_ladder[1:],
            extra=rung(one_prefix, i_lakha, 4, digit),
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
        graph_crores, graph_ten_crores = band(
            exact_n(8), i_koti, i_kotlu, i_kotlu_pl, i_koti_sp, 7, crore_ladder, ten_oka=i_koti_sp
        )

        hundred_crore = U(
            exact_n(3),
            hp_ten + digit_x1,
            hp_one + teens_ties,
            suffix(digit_x1, i_vandalu, 2),
            rung(digit_x1, i_vandalu, 1, digit_x1),
            rung(digit_x1, i_vandalu, 0, teens_ties),
        )
        hundred_one = U(
            hp_one + pynutil.delete(NEMO_ALL_ZERO) + one_as_oka,
            digit_x1 + pynutil.delete(NEMO_ALL_ZERO) + hundreds_before_one,
        )
        thou_crore_ladder = [(2, digit_x1), (1, teens_ties), (0, hundred_crore)]
        thousand_one = oka_count(i_thou, i_thous, [(2, one_as_oka), (0, hundred_one)])
        thousand_crore, ten_thousand_crore = band(
            exact_n(4), i_thou, i_thous, i_thous, i_thou_sp, 3, thou_crore_ladder
        )
        graph_hundred_crores = U(crore_of(hundred_one, hundred_crore), rung(hundred_crore, i_kotlu, 0, graph_crores))
        graph_thousand_crores = crore_of(thousand_one, U(thousand_crore, ten_thousand_crore))
        graph_ten_thousand_crores = crore_of(thousand_one, ten_thousand_crore)

        lakh_crore_ladder = [
            (4, digit_x1),
            (3, teens_ties),
            (2, hundred_crore),
            (1, graph_thousands),
            (0, ten_thousand_crore),
        ]
        lakh_one = oka_count(
            i_lakh,
            i_lakhs,
            [(4, one_as_oka), (2, hundred_one), (1, exact_n(4, thousand_one)), (0, exact_n(5, thousand_one))],
        )
        lakh_crore, ten_lakh_crore = band(exact_n(6), i_lakh, i_lakhs, i_lakhs, i_lakh_sp, 5, lakh_crore_ladder)
        graph_lakh_crores = crore_of(lakh_one, U(lakh_crore, ten_lakh_crore))

        ten_lakh_rem = ties_group(i_lakhs, i_lakhs, lakh_ladder, i_lakhs, i_lakhs, 5)
        koti_ladder = [
            (6, digit),
            (5, teens_ties),
            (4, graph_hundreds),
            (3, graph_thousands),
            (2, graph_ten_thousands),
            (1, graph_lakhs),
            (0, ten_lakh_rem),
        ]
        crore_one = oka_count(
            i_koti,
            i_kotlu,
            [(6, one_as_oka), (4, hundred_one), (1, exact_n(6, lakh_one)), (0, exact_n(7, lakh_one))],
        )
        except_one_ladder = [(2, digit_x1), (1, teens_ties_x1)]
        hundred_amt = U(
            exact_n(3),
            hp_ten + digit_x1,
            hp_one + teens_ties_x1,
            suffix(digit_x1, i_vandalu, 2),
            rung(digit_x1, i_vandalu, 1, digit_x1),
        )
        thousand_amt = U(*band(exact_n(4), i_thou, i_thous, i_thous, i_thou_sp, 3, except_one_ladder))
        ten_thousand_amt = ties_group(i_thous, i_thous, except_one_ladder + [(0, hundred_amt)], i_thou_sp, i_thous, 3)
        crore_one_10 = rung(hundred_amt, i_kotlu, 6, one_as_oka).optimize()
        crore_one_11 = rung(thousand_amt, i_kotlu, 6, one_as_oka).optimize()
        crore_one_12 = rung(ten_thousand_amt, i_kotlu, 6, one_as_oka).optimize()

        ten_lakh_crore_count = prefer(
            U(
                rung(one_prefix, i_koti, 0, ten_lakh_rem),
                rung(digit_x1, i_kotlu, 0, ten_lakh_rem),
                rung(teens_ties_oka, i_koti_sp, 0, ten_lakh_rem),
                rung(teens_ties_x1, i_kotlu, 0, ten_lakh_rem),
            ),
            U(graph_crores, graph_ten_crores),
        ).optimize()
        graph_ten_lakh_crores = prefer(
            kotlu(exact_n(8, crore_one) + i_koti_sp), kotlu(ten_lakh_crore_count)
        ).optimize()
        graph_crore_crores = prefer(
            kotlu(exact_n(9, crore_one) + i_koti_sp),
            kotlu(ties_group(i_koti_sp, i_kotlu, koti_ladder, i_koti_sp, i_kotlu, 7)),
        ).optimize()

        graph_ten_crore_crores = kotlu(
            prefer(
                oka_koti(hundred_one, crore_one_10, bare=(crore_one_10,)),
                group(hundred_crore, i_kotlu, koti_ladder, i_kotlu, 7),
            ).optimize()
        )
        hcc_oka = group(teens_ties_oka, i_thous, thou_crore_ladder, i_thou_sp, 3)
        hcc_other = U(thousand_crore, group(teens_ties_x1, i_thous, thou_crore_ladder, i_thous, 3))
        graph_hundred_crore_crores = kotlu(
            prefer(
                oka_koti(
                    thousand_one,
                    hcc_oka,
                    crore_one_12,
                    crore_one_11,
                    crore_one_10,
                    bare=(crore_one_12, crore_one_11),
                    pad=((crore_one_10, 1),),
                ),
                group(hcc_other, i_kotlu, koti_ladder, i_kotlu, 7),
            ).optimize()
        )

        graph = U(
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
            U(
                pynini.closure(NEMO_ALL_DIGIT, 1, 2) + pynini.closure(sep + two) + sep + three,
                pynini.closure(NEMO_ALL_DIGIT, 1, 3) + pynini.closure(sep + three, 1),
            ),
            graph,
        ).optimize()

        final = U(graph, leading_zeros, grouped)
        minus = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)
        self.final_graph = final.optimize()
        self.fst = self.add_tokens(minus + pynutil.insert('integer: "') + self.final_graph + pynutil.insert('"'))
