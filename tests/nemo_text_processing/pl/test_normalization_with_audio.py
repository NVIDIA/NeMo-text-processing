# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

from ..utils import get_test_cases_multiple


class TestNormalizeWithAudio:
    normalizer = NormalizerWithAudio(input_case="cased", lang="pl", cache_dir=None, post_process=False)

    @parameterized.expand(
        get_test_cases_multiple("pl/data_text_normalization/test_cases_normalize_with_audio.txt")
    )
    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_normalization_with_audio(self, test_input, expected):
        predictions = self.normalizer.normalize(test_input, n_tagged=100, punct_post_process=False)
        for option in expected:
            assert option in predictions
