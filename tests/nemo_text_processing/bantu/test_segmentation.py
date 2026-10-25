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

"""Kernel sentence segmentation with the made-up test profile: numbers inside sentences."""

import pytest
from parameterized import parameterized

from .toy_profile import TOY_PROFILE
from .utils import best_tags, build_normalizer

NORMALIZER = build_normalizer(TOY_PROFILE)


def check(spoken, expected):
    assert len(best_tags(NORMALIZER, spoken)) == 1
    assert NORMALIZER.inverse_normalize(spoken, verbose=False) == expected


class TestBantuKernelSegmentation:
    # The join word vo is also an ordinary word ("in"): followed by an ordinary word,
    # it ends the number instead of joining a remainder.
    @parameterized.expand(
        [
            ("bakopa bapi vo kasa", "bakopa 2 vo kasa"),
            ("zikopa pideka vo kasa", "zikopa 20 vo kasa"),
            ("rana bakopa bako vo kasa lena", "rana bakopa 3 vo kasa lena"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_join_word_as_ordinary_word(self, spoken, expected):
        check(spoken, expected)

    # Several numbers and plain words in one sentence.
    @parameterized.expand(
        [
            ("rana bakopa bapi lena zikopa ziko", "rana bakopa 2 lena zikopa 3"),
            ("bakopa bapi zikopa ziko", "bakopa 2 zikopa 3"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_numbers_in_sentences(self, spoken, expected):
        check(spoken, expected)

    # No counted noun, no currency: number words stay as words.
    @parameterized.expand(
        [
            ("bapi", "bapi"),
            ("rana bapi lena", "rana bapi lena"),
            ("kasa vo rana", "kasa vo rana"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_unchanged(self, spoken, expected):
        check(spoken, expected)
