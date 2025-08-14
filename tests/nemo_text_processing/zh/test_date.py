# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.normalize import Normalizer

from tests.nemo_text_processing.utils import CACHE_DIR, assert_projecting_output, parse_test_case_file


class TestDate:

    normalizer = Normalizer(lang='zh', cache_dir=CACHE_DIR, overwrite_cache=False, input_case='cased')
    normalizer_project = Normalizer(
        lang='zh', project_input=True, cache_dir=CACHE_DIR, overwrite_cache=False, input_case='cased'
    )
    inverse_normalizer = InverseNormalizer(lang='zh', cache_dir=CACHE_DIR, overwrite_cache=False)
    inverse_normalizer_project = InverseNormalizer(
        lang='zh', project_input=True, cache_dir=CACHE_DIR, overwrite_cache=False
    )

    @parameterized.expand(parse_test_case_file('zh/data_text_normalization/test_cases_date.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_date(self, test_input, expected):
        pred = self.normalizer.normalize(test_input)
        assert pred == expected

    @parameterized.expand(parse_test_case_file('zh/data_text_normalization/test_cases_date.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_date_project_input(self, test_input, expected):
        pred = self.normalizer_project.normalize(test_input)
        assert_projecting_output(pred, expected, test_input)

    @parameterized.expand(parse_test_case_file('zh/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_date(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected

    @parameterized.expand(parse_test_case_file('zh/data_inverse_text_normalization/test_cases_date.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_date_project_input(self, test_input, expected):
        pred = self.inverse_normalizer_project.inverse_normalize(test_input, verbose=False)
        assert_projecting_output(pred, expected, test_input)
