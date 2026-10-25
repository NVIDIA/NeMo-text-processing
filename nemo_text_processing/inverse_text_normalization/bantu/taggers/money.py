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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CurrencyContext,
    agreement_context,
)
from nemo_text_processing.inverse_text_normalization.bantu.taggers.number import (
    build_cardinal,
    build_role,
    integer_domain,
)
from nemo_text_processing.inverse_text_normalization.bantu.taggers.scale import build_scale_amount
from nemo_text_processing.inverse_text_normalization.bantu.taggers.sentence import CUE_ONLY, IMPLIED, run
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


def build_amount(profile: CardinalProfile, currency: CurrencyContext) -> pynini.Fst:
    """Digits for an amount after this currency phrase: agreeing numbers, licensed roles and counted scales."""
    amount = build_cardinal(profile, agreement_context(profile, currency.agreement_class))
    for role in currency.roles:
        amount |= build_role(profile, role) @ integer_domain(profile.max_value)
    for scale in profile.counted_scales:
        amount |= build_scale_amount(profile, scale)
    return amount.optimize()


def money_token(amount: pynini.Fst, code: str, marker: str = "") -> pynini.Fst:
    return run(
        pynutil.insert(marker + 'tokens { money { integer_part: "')
        + amount
        + pynutil.insert(f'" currency: "{code}" }} }}')
    )


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        koin mina epi -> tokens { money { integer_part: "20000" currency: "XTS" } }

    An amount after a spoken currency phrase is always money. A counted scale may
    also imply a currency without one (Luganda mitwalo ebiri, 20,000 UGX):
    - bare scale heads (mitwalo) are money anywhere, or only after a money cue
      when the scale's implied_anywhere is not set;
    - heads with the augment (emitwalo, akakadde) are money only after a money cue,
      or also at the start of the input when counting_money_at_start is set.
    Implied amounts carry a marker, checked and removed by only_with_money_signal
    once the whole sentence is tagged.

    Args:
        profile: the language profile
    """

    def __init__(self, profile: CardinalProfile):
        super().__init__(name="money", kind="classify")

        graphs = []
        for currency in profile.currencies:
            graphs.append(
                pynutil.delete(currency.phrase + " ") + money_token(build_amount(profile, currency), currency.code)
            )

        self.has_implied_currency = False
        for scale in profile.counted_scales:
            if not scale.implied_currency:
                continue
            self.has_implied_currency = True
            bare = build_scale_amount(profile, scale, "bare")
            graphs.append(money_token(bare, scale.implied_currency, "" if scale.implied_anywhere else IMPLIED))
            if scale.counting_plural or scale.counting_ones or scale.money_only_ones:
                augmented = build_scale_amount(profile, scale, "counting_money")
                marker = IMPLIED if scale.counting_money_at_start else CUE_ONLY
                graphs.append(money_token(augmented, scale.implied_currency, marker))

        self.fst = pynini.union(*graphs).optimize() if graphs else pynini.Fst()
