# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Counted scales: a scale noun, its count (1-999), then an optional smaller remainder.

The remainder may itself be a smaller counted scale:
obukadde bubiri mu mitwalo ataano = 2 x 1,000,000 + 50 x 10,000.

Example (Luganda): mitwalo abiri mu etaano mu nkumi ttaano = 25 x 10,000 + 5,000.
Unit words in the count agree with the scale noun; the remainder agrees with what
is counted. A remainder must be at least 10, so a single unit after the join always
belongs to the count (6-9 have the same form in every class).
"""
from typing import Optional, Tuple

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CountedScale,
    NounContext,
    agreement_context,
)
from nemo_text_processing.inverse_text_normalization.bantu.taggers.number import (
    DIGIT,
    NONZERO,
    build_cardinal,
    integer_domain,
    mapping,
)


def build_count(profile: CardinalProfile, scale: CountedScale) -> Tuple[pynini.Fst, pynini.Fst]:
    """Digits for a plural count: 02-99, or 100-999 when the scale has count_hundreds.

    Returns (count, round_hundreds): round_hundreds is the part of count that is
    hundreds alone (obukadde kikumi), which a remainder below 100 may not follow.
    """
    context = agreement_context(profile, scale.count_class)
    final_units = context.final_units + scale.count_ones
    units = mapping(final_units)  # 1-9
    plural_units = mapping(tuple((f, v) for f, v in context.atomic_units if v != "10"))
    tens = mapping(scale.tens)
    higher_tens = mapping(tuple((f, v) for f, v in scale.tens if v != "1"))
    join = profile.places[0].remainder_join
    teen_units = []
    for form, value in final_units:
        if form[0] in profile.vowels:
            spoken = [" " + prefix + form for prefix in scale.contracted_teen_joins]
        else:
            spoken = [profile.teen_join + form]
        teen_units.extend(pynini.cross(s, value) for s in spoken)
    teens = pynini.cross(profile.ten, "1") + pynini.union(*teen_units)
    two_digits = (
        pynutil.insert("0") + plural_units
        | tens + pynutil.insert("0")
        | teens
        | higher_tens + pynutil.delete(join) + units
    ).optimize()
    if not scale.count_hundreds:
        return two_digits, pynini.Fst()
    # obukadde kikumi mu ataano = 150: hundreds, then an optional mu + 01-99.
    below_hundred = two_digits | pynutil.insert("0") + units
    hundreds = mapping(scale.count_hundreds)
    count = two_digits | hundreds + pynutil.delete(join) + below_hundred
    return count.optimize(), (hundreds + pynutil.insert("00")).optimize()


HEADS = ("bare", "counting", "counting_money", "all")


def build_scale_amount(
    profile: CardinalProfile, scale: CountedScale, heads: str = "all", remainder_context: Optional[NounContext] = None
) -> pynini.Fst:
    """Digits for a counted-scale amount, bounded by the profile maximum.

    heads picks the spoken heads: "bare" (mitwalo, kakadde kamu), "counting" (with
    the augment: emitwalo, akakadde kamu), "counting_money" (counting + money-only
    ones such as akakadde), or "all". Empty if the scale has no such heads.
    remainder_context sets the agreement after the join (default: remainder_class),
    e.g. the counted noun in "ebitabo obukadde bubiri".
    """
    if heads not in HEADS:
        raise ValueError(f"unknown heads {heads!r}")
    join = profile.places[0].remainder_join
    count, round_hundreds = build_count(profile, scale)
    ones, plurals = [], []
    if heads in ("bare", "all"):
        ones.append(f"{scale.singular} {scale.singular_count}")
        plurals.append(scale.plural)
    if heads != "bare":
        ones.extend(scale.counting_ones)
        plurals.extend([scale.counting_plural] if scale.counting_plural else [])
    if heads in ("counting_money", "all"):
        ones.extend(scale.money_only_ones)
    if not ones and not plurals:
        return pynini.Fst()
    plural = pynini.union(*(pynutil.delete(p + " ") for p in plurals)) if plurals else pynini.Fst()
    head = plural + count
    if ones:
        head |= pynini.union(*(pynini.cross(one, "01") for one in ones))
    width = scale.power

    def padded(smallest_digits):
        return pynini.union(
            *(
                pynutil.insert("0" * (width - k)) + NONZERO + pynini.closure(DIGIT, k - 1, k - 1)
                for k in range(smallest_digits, width + 1)
            )
        )

    context = remainder_context or agreement_context(profile, scale.remainder_class)
    remainder = build_cardinal(profile, context)
    for smaller in profile.counted_scales:
        if smaller.power < scale.power:
            remainder |= build_scale_amount(profile, smaller, "all", context)
    amount = head + (pynutil.insert("0" * width) | pynutil.delete(join) + (remainder @ padded(2)))
    # After round hundreds, tens/units words continue the count:
    # obukadde lwenda mu kyenda mu mwenda = 999 million, never 900 million + 99.
    amount |= plural + round_hundreds + (pynutil.insert("0" * width) | pynutil.delete(join) + (remainder @ padded(3)))
    canonical = pynutil.delete(pynini.closure("0")) + NONZERO + pynini.closure(DIGIT)
    return (amount @ canonical @ integer_domain(profile.max_value)).optimize()


def build_noun_scale_amount(profile: CardinalProfile, context: NounContext) -> pynini.Fst:
    """Scale amounts that may count a noun (ebitabo obukadde bubiri), agreeing with it."""
    amounts = [
        build_scale_amount(profile, scale, "counting", context)
        for scale in profile.counted_scales
        if scale.after_counted_nouns and context.compounds
    ]
    return pynini.union(*amounts).optimize() if amounts else pynini.Fst()
