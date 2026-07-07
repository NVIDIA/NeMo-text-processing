# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
import pytest
from parameterized import parameterized

from nemo_text_processing.inverse_text_normalization.te.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.verbalize_final import VerbalizeFinalFst

from ..utils import parse_test_case_file


class TestCardinal:
    tagger = ClassifyFst(overwrite_cache=True).fst
    verbalizer = VerbalizeFinalFst().fst

    @parameterized.expand(parse_test_case_file('te/data_inverse_text_normalization/test_cases_cardinal.txt'))
    #@pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        tagged = pynini.shortestpath(test_input @ self.tagger).string()
        pred = pynini.shortestpath(tagged @ self.verbalizer).string()
        assert pred == expected