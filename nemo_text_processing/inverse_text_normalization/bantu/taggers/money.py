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

"""Currency-context amounts: reuse cardinal composition, then add licensed roles."""
import pynini

from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CurrencyContext,
    agreement_context,
)
from nemo_text_processing.inverse_text_normalization.bantu.taggers.cardinal import (
    build_cardinal,
    build_role,
    integer_domain,
)
from nemo_text_processing.inverse_text_normalization.bantu.taggers.scale import build_scale_amount


def build_amount(profile: CardinalProfile, currency: CurrencyContext) -> pynini.Fst:
    """Digits for an amount after this currency phrase; no Python arithmetic."""
    amount = build_cardinal(profile, agreement_context(profile, currency.agreement_class))
    for role in currency.roles:
        amount |= build_role(profile, role) @ integer_domain(profile.max_value)
    for scale in profile.counted_scales:
        amount |= build_scale_amount(profile, scale)
    return amount.optimize()
