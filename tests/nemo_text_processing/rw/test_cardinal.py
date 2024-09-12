# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
# Copyright 2015 and onwards Google, Inc.
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

from ..utils import CACHE_DIR, parse_test_case_file


class TestCardinal:

    
    normalizer_rw = Normalizer(
        input_case='cased', lang='rw', cache_dir=CACHE_DIR, overwrite_cache=False, post_process=True
    )
    

    @parameterized.expand(parse_test_case_file('rw/data_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        pred = self.normalizer_rw.normalize(test_input, verbose=False, punct_post_process=False)
        assert pred == expected, f"input: {test_input}"
        print(pred)

       