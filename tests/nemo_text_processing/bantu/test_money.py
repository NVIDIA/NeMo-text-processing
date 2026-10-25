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

"""Kernel money with the made-up test profile: currency context, counted scales, implied currency."""

import pytest
from parameterized import parameterized

from .toy_profile import TOY_PROFILE
from .utils import best_tags, build_normalizer

NORMALIZER = build_normalizer(TOY_PROFILE)


def check(spoken, expected):
    assert len(best_tags(NORMALIZER, spoken)) == 1
    assert NORMALIZER.inverse_normalize(spoken, verbose=False) == expected


class TestBantuKernelMoney:
    # A currency phrase sets the agreement and licenses extra readings: mapi is 20 only after koin.
    @parameterized.expand(
        [
            ("koin zipi", "2 XTS"),
            ("koin mapi", "20 XTS"),
            ("makopa mapi", "makopa 2"),
            ("koin milo vo hanto", "1,100 XTS"),
            ("koin", "koin"),
            ("koin kasa", "koin kasa"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_currency_context(self, spoken, expected):
        check(spoken, expected)

    # Ten-thousands: mina + count (agreeing with mina's class) + optional vo remainder.
    @parameterized.expand(
        [
            ("muna ema", "10,000 XTS"),
            ("mina epi", "20,000 XTS"),
            ("mina pideka vo efu vo milo fu", "255,000 XTS"),
            ("koin mina epi", "20,000 XTS"),
            ("mina zipi", "mina zipi"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_counted_scale(self, spoken, expected):
        check(spoken, expected)

    # A teen join before a vowel contracts (k'epi or kepi); the full form is not accepted.
    @parameterized.expand(
        [
            ("mina deka k'epi", "120,000 XTS"),
            ("mina deka kepi", "120,000 XTS"),
            ("mina deka ke epi", "mina deka ke epi"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_contracted_teens(self, spoken, expected):
        check(spoken, expected)

    # Millions: one forms, teens with a "one", hundreds in the count, and a smaller
    # scale as the remainder. Tens/units after round hundreds continue the count.
    @parameterized.expand(
        [
            ("kamega kaima", "1,000,000 XTS"),
            ("bumega bapi", "2,000,000 XTS"),
            ("bumega deka ke kaima", "11,000,000 XTS"),
            ("bumega hanto vo fudeka", "150,000,000 XTS"),
            ("bumega hanto nona vo nonadeka vo nona", "999,000,000 XTS"),
            ("bumega hanto vo milo fu", "100,005,000 XTS"),
            ("bumega bapi vo mina fudeka", "2,500,000 XTS"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_larger_scale(self, spoken, expected):
        check(spoken, expected)

    # Implied currency: bare mina is always money; word + emina counts that word,
    # unless the word is a money cue; emina with nothing before it stays words.
    @parameterized.expand(
        [
            ("kasa mina epi", "kasa 20,000 XTS"),
            ("kasa emina epi", "kasa 20,000"),
            ("lipoka emina epi", "lipoka 20,000 XTS"),
            ("dalo emina epi", "dalo 20,000 XTS"),
            ("olipon emina epi", "olipon 20,000"),
            ("emina epi", "emina epi"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_implied_currency(self, spoken, expected):
        check(spoken, expected)

    # Millions: augmented forms are also money at the start of input, count a listed
    # or unlisted noun before them, and need "one" (kaima) to count a single million.
    @parameterized.expand(
        [
            ("obumega bapi", "2,000,000 XTS"),
            ("akamega", "1,000,000 XTS"),
            ("akamega kaima", "1,000,000 XTS"),
            ("kasa bumega bapi", "kasa 2,000,000 XTS"),
            ("kasa obumega bapi", "kasa 2,000,000"),
            ("kasa akamega kaima", "kasa 1,000,000"),
            ("kasa akamega", "kasa akamega"),
            ("bakopa obumega bapi", "bakopa 2,000,000"),
            ("bakopa akamega kaima", "bakopa 1,000,000"),
            ("lipoka obumega bapi", "lipoka 2,000,000 XTS"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_augmented_larger_scale(self, spoken, expected):
        check(spoken, expected)

    # A held-back word continuing the number keeps the whole phrase unchanged.
    @parameterized.expand(
        [
            ("mina epi halfu", "mina epi halfu"),
            ("lipoka mina epi halfu", "lipoka mina epi halfu"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_held_back(self, spoken, expected):
        check(spoken, expected)
