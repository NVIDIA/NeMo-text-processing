<<<<<<< HEAD
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
=======
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
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

<<<<<<< HEAD
import pytest
from nemo_text_processing.text_normalization.normalize import Normalizer
=======

import pytest
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
from parameterized import parameterized

from ..utils import CACHE_DIR, parse_test_case_file


<<<<<<< HEAD
class TestChar:
    normalizer = Normalizer(lang='it', cache_dir=CACHE_DIR, overwrite_cache=False, input_case='cased')

<<<<<<< HEAD:tests/nemo_text_processing/zh/test_word.py
    @parameterized.expand(parse_test_case_file('zh/data_text_normalization/test_cases_word.txt'))
=======
    @parameterized.expand(parse_test_case_file('it/data_text_normalization/test_cases_cardinal.txt'))
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6:tests/nemo_text_processing/it/test_cardinal.py
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_char(self, test_input, expected):
        preds = self.normalizer.normalize(test_input)
        assert expected == preds
=======
class TestWord:
    inverse_normalizer = InverseNormalizer(lang='zh', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('zh/data_inverse_text_normalization/test_cases_word.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
