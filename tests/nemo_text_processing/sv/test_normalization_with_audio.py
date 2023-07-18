# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from parameterized import parameterized

from ..utils import CACHE_DIR, RUN_AUDIO_BASED_TESTS, get_test_cases_multiple


class TestNormalizeWithAudio:
    normalizer_with_audio = (
        NormalizerWithAudio(input_case='cased', lang='sv', cache_dir=CACHE_DIR, overwrite_cache=False)
        if RUN_AUDIO_BASED_TESTS
        else None
    )

    @parameterized.expand(get_test_cases_multiple('sv/data_text_normalization/test_cases_normalize_with_audio.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        if self.normalizer_with_audio:
            pred = self.normalizer_with_audio.normalize(test_input, n_tagged=150, punct_post_process=False)
            assert len(set(pred).intersection(set(expected))) == len(
                expected
            ), f'missing: {set(expected).difference(set(pred))}'
