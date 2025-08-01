# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.normalize import Normalizer

from tests.nemo_text_processing.utils import parse_test_case_file


class TestRange:
    normalizer = Normalizer(input_case='cased', lang='vi', cache_dir=None, overwrite_cache=True)

    @parameterized.expand(parse_test_case_file("vi/data_text_normalization/test_cases_range.txt"))
    @pytest.mark.run_only_on('CPU')
    def test_norm(self, test_input, expected):
        pred = self.normalizer.normalize(test_input)
        assert pred == expected, f"input: {test_input} assert {pred} == {expected}"
