# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.it.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.it.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.normalize import Normalizer

from ..utils import CACHE_DIR, parse_test_case_file


@pytest.fixture(scope='module', params=[True, False])
def time_verbalizer(request):
    deterministic = request.param
    cardinal = CardinalFst(deterministic=deterministic)
    return deterministic, TimeFst(cardinal_tagger=cardinal, deterministic=deterministic).fst


class TestChar:
    normalizer = Normalizer(lang='it', cache_dir=CACHE_DIR, overwrite_cache=False, input_case='cased')

    @parameterized.expand(parse_test_case_file('it/data_text_normalization/test_cases_time.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_char(self, test_input, expected):
        preds = self.normalizer.normalize(test_input, punct_post_process=True)
        assert preds == (expected[0] if isinstance(expected, list) else expected)

    @pytest.mark.parametrize(
        'tagged, expected, alternative',
        [
            ('time { hours: "12" minutes: "15" }', 'dodici e quindici minuti', 'dodici e un quarto'),
            ('time { hours: "12" minutes: "30" }', 'dodici e trenta minuti', 'dodici e mezza'),
            (
                'time { hours: "17" minutes: "15" seconds: "26" preserve_order: true }',
                'diciassette e quindici minuti e ventisei secondi',
                'diciassette e un quarto e ventisei secondi',
            ),
            (
                'time { hours: "15" minutes: "30" seconds: "30" preserve_order: true }',
                'quindici e trenta minuti e trenta secondi',
                'quindici e mezza e trenta secondi',
            ),
        ],
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_time_verbalizer_priorities(self, time_verbalizer, tagged, expected, alternative):
        deterministic, grammar = time_verbalizer
        lattice = pynini.compose(tagged, grammar)
        output_weights = {}
        for _, output, weight in lattice.paths().items():
            output = ' '.join(output.split())
            output_weights[output] = min(output_weights.get(output, float('inf')), float(weight))
        assert set(output_weights) == {expected, alternative}
        if deterministic:
            assert output_weights[expected] < output_weights[alternative]
        else:
            assert output_weights[expected] == output_weights[alternative]
