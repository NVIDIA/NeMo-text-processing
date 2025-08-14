# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from tests.nemo_text_processing.utils import CACHE_DIR, parse_test_case_file, assert_projecting_output

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer


class TestElectronic:

    normalizer = NormalizerWithAudio(input_case='cased', lang='ru', cache_dir=CACHE_DIR)
    normalizer_project = Normalizer(input_case='cased', lang='ru', deterministic=False, project_input=True, cache_dir=CACHE_DIR)
    inverse_normalizer = InverseNormalizer(lang='ru', cache_dir=CACHE_DIR)
    inverse_normalizer_project = InverseNormalizer(lang='ru', project_input=True, cache_dir=CACHE_DIR)
    N_TAGGED = 100

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_electronic(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_electronic_project_input(self, expected, test_input):
        pred = self.normalizer_project.normalize(test_input)
        assert_projecting_output(pred, expected, test_input)

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_electronic(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert expected == pred

    @parameterized.expand(parse_test_case_file('ru/data_inverse_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_electronic_project_input(self, test_input, expected):
        pred = self.inverse_normalizer_project.inverse_normalize(test_input, verbose=False)
        assert_projecting_output(pred, expected, test_input)
