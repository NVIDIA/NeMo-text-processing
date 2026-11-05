# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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

"""Kernel behaviour that depends on the profile itself: validation, optional features, caching."""

from dataclasses import replace

import pytest

from nemo_text_processing.inverse_text_normalization.bantu.graph_utils import integer_domain
from nemo_text_processing.inverse_text_normalization.bantu.profile import MoneyCues, Place
from nemo_text_processing.inverse_text_normalization.bantu.taggers.cardinal import build_role
from nemo_text_processing.inverse_text_normalization.bantu.taggers.scale import build_scale_amount
from nemo_text_processing.inverse_text_normalization.bantu.taggers.tokenize_and_classify import build_classifier

from .toy_profile import TOY_PROFILE
from .utils import build_normalizer


class TestBantuKernelProfile:
    # A broken profile is rejected when the grammar is built, with a clear error.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invalid_profiles_are_rejected(self):
        with pytest.raises(ValueError, match="no compound agreement context"):
            build_classifier(
                replace(TOY_PROFILE, currencies=(replace(TOY_PROFILE.currencies[0], agreement_class="99"),))
            )
        with pytest.raises(ValueError, match="consecutive ascending"):
            build_classifier(replace(TOY_PROFILE, places=TOY_PROFILE.places[1:]))
        with pytest.raises(ValueError, match="single nonzero digits"):
            build_classifier(
                replace(TOY_PROFILE, places=(Place(1, (("pideka", "20"),), " vo "),) + TOY_PROFILE.places[1:])
            )
        with pytest.raises(ValueError, match="positive integer"):
            integer_domain(0)
        with pytest.raises(KeyError):
            build_role(TOY_PROFILE, "no-such-role")
        with pytest.raises(ValueError, match="unknown heads"):
            build_scale_amount(TOY_PROFILE, TOY_PROFILE.counted_scales[0], "no-such-heads")

    # Optional features can be left out: no money cues, and scales with no implied currency.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_profile_without_money_cues_or_implied_currency(self):
        plain = replace(
            TOY_PROFILE,
            money_cues=MoneyCues(),
            counted_scales=tuple(replace(s, implied_currency="") for s in TOY_PROFILE.counted_scales),
        )
        normalizer = build_normalizer(plain)
        assert normalizer.inverse_normalize("mina epi", verbose=False) == "mina epi"
        assert normalizer.inverse_normalize("koin mina epi", verbose=False) == "20,000 XTS"
        assert normalizer.inverse_normalize("kasa emina epi", verbose=False) == "kasa 20,000"
        assert normalizer.inverse_normalize("lipoka emina epi", verbose=False) == "lipoka 20,000"

    # Implied currency without any money cues: only the start-of-input and bare-form rules apply.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_implied_currency_without_money_cues(self):
        normalizer = build_normalizer(replace(TOY_PROFILE, money_cues=MoneyCues()))
        assert normalizer.inverse_normalize("obumega bapi", verbose=False) == "2,000,000 XTS"
        assert normalizer.inverse_normalize("kasa mina epi", verbose=False) == "kasa 20,000 XTS"
        assert normalizer.inverse_normalize("lipoka emina epi", verbose=False) == "lipoka 20,000"

    # A scale without counting heads simply has no counting reading.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_scale_without_counting_heads(self):
        bare_only = replace(TOY_PROFILE.counted_scales[0], counting_plural="", counting_ones=())
        assert build_scale_amount(TOY_PROFILE, bare_only, "counting").num_states() == 0

    # Money reuses the cardinal's graphs, as in other NeMo languages.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_money_takes_the_cardinal(self):
        from nemo_text_processing.inverse_text_normalization.bantu.graph_utils import join_aware_runs
        from nemo_text_processing.inverse_text_normalization.bantu.taggers.cardinal import CardinalFst
        from nemo_text_processing.inverse_text_normalization.bantu.taggers.money import MoneyFst

        _, ordinary_word, _ = join_aware_runs(TOY_PROFILE.number_words(), TOY_PROFILE.join_words())
        cardinal = CardinalFst(TOY_PROFILE, ordinary_word)
        assert cardinal.profile is TOY_PROFILE
        assert set(cardinal.by_class) == {"2", "4", "6", "8"}
        assert set(cardinal.roles) == {"short-tens"}
        money = MoneyFst(cardinal)
        assert money.has_implied_currency
        assert money.fst.num_states() > 0
