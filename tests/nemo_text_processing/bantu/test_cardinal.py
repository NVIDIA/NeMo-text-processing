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

"""Kernel cardinals with the made-up test profile: agreement, composition, malformed input."""

import pytest
from parameterized import parameterized

from .toy_profile import TOY_PROFILE
from .utils import best_tags, build_normalizer

NORMALIZER = build_normalizer(TOY_PROFILE)


def check(spoken, expected):
    # Exactly one best reading: the kernel never leaves a tie for NeMo to pick from.
    assert len(best_tags(NORMALIZER, spoken)) == 1
    assert NORMALIZER.inverse_normalize(spoken, verbose=False) == expected


class TestBantuKernelCardinal:
    # Number words agree with the counted noun's class; a wrong class stays unchanged.
    @parameterized.expand(
        [
            ("bakopa bapi", "bakopa 2"),
            ("zikopa zipi", "zikopa 2"),
            ("makopa mapi", "makopa 2"),
            ("kopa ima", "kopa 1"),
            ("bakopa zipi", "bakopa zipi"),
            ("bakopa ima", "bakopa ima"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_agreement(self, spoken, expected):
        check(spoken, expected)

    # 6-9 and ten have one form in every class.
    @parameterized.expand(
        [
            ("bakopa sei", "bakopa 6"),
            ("zikopa nona", "zikopa 9"),
            ("makopa deka", "makopa 10"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_invariant_units(self, spoken, expected):
        check(spoken, expected)

    # Teens join with ke; tens, hundreds and thousands join with vo; a trailing one agrees.
    @parameterized.expand(
        [
            ("zikopa deka ke zipi", "zikopa 12"),
            ("bakopa pideka vo baima", "bakopa 21"),
            ("zikopa hanto pi vo kodeka vo ziko", "zikopa 233"),
            ("bakopa milo", "bakopa 1,000"),
            ("bakopa milo vo hanto", "bakopa 1,100"),
            ("zikopa milo nona vo hanto nona vo nonadeka vo nona", "zikopa 9,999"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_composition(self, spoken, expected):
        check(spoken, expected)

    # Profile extras: a fixed phrase, a special teen form and a spelling variant.
    @parameterized.expand(
        [
            ("bakopa tausi", "bakopa 1,000"),
            ("makopa dekafu", "makopa 15"),
            ("bakopa piideka", "bakopa 20"),
            ("bakopa piideka vo baima", "bakopa 21"),
            ("koin piideka", "20 XTS"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_profile_extras(self, spoken, expected):
        check(spoken, expected)

    # A malformed run of number words is left unchanged as a whole, never partly converted.
    @parameterized.expand(
        [
            ("bakopa hanto vo milo", "bakopa hanto vo milo"),
            ("bakopa deka ke", "bakopa deka ke"),
            ("bakopa bapi bako", "bakopa bapi bako"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_malformed(self, spoken, expected):
        check(spoken, expected)
