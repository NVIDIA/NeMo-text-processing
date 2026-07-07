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

import pytest
from parameterized import parameterized

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

from ..utils import CACHE_DIR, parse_test_case_file


class TestCardinal:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestOrdinal:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestDecimal:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestTime:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_time.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestDate:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestMeasure:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_measure.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestMoney:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_money.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestElectronic:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestTelephone:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_telephone.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestFraction:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_fraction.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


class TestWord:
    inverse_normalizer = InverseNormalizer(lang='el', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('el/data_inverse_text_normalization/test_cases_word.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected
