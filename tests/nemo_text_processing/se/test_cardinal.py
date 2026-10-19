# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.se.taggers.cardinal import CardinalFst

from ..utils import CACHE_DIR, parse_test_case_file


class TestCardinal:
    normalizer = Normalizer(input_case='cased', lang='se', cache_dir=CACHE_DIR, overwrite_cache=False)

    @parameterized.expand(parse_test_case_file('se/data_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        pred = self.normalizer.normalize(test_input, verbose=False)
        assert pred == expected


class TestCardinalNonDeterministic:
    cardinal = CardinalFst(deterministic=False)

    @parameterized.expand(
        [
            ("20", "ill_sg", "guovttelohkái"),
            ("20", "loc_sg", "guovttelogis"),
            ("200", "ill_sg", "guovttečuohtái"),
            ("200", "loc_sg", "guovttečuođis"),
            ("201", "ill_sg", "guoktečuođioktii"),
            ("201", "loc_sg", "guoktečuođiovttas"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_cased_compound(self, test_input, case, expected):
        assert expected in rewrite.rewrites(test_input, self.cardinal.graphs[case])

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_alternative_readings(self):
        assert {"akta", "okta"} <= set(rewrite.rewrites("1", self.cardinal.graph))
        assert {"guhtti", "guhttii"} <= set(rewrite.rewrites("6:i", self.cardinal.graph))
