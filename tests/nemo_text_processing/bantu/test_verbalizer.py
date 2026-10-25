# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Kernel verbalizers on their own: formatting only, values are resolved by the classifier."""

import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.verbalize_final import VerbalizeFinalFst

from .toy_profile import TOY_PROFILE

VERBALIZER = VerbalizeFinalFst(TOY_PROFILE).fst


class TestBantuKernelVerbalizer:
    @parameterized.expand(
        [
            ('tokens { cardinal { integer: "2" } }', "2"),
            ('tokens { cardinal { integer: "999999999" } }', "999,999,999"),
            ('tokens { money { integer_part: "25000" currency: "XTS" } }', "25,000 XTS"),
            ('tokens { name: "kasa" } tokens { cardinal { integer: "1000" } }', "kasa 1,000"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_formatting(self, tokens, expected):
        assert rewrite.top_rewrite(tokens, VERBALIZER) == expected

    # Values outside 1..max_value, non-canonical digits and lowercase codes are rejected.
    @parameterized.expand(
        [
            ('tokens { cardinal { integer: "0" } }',),
            ('tokens { cardinal { integer: "02" } }',),
            ('tokens { cardinal { integer: "1000000000" } }',),
            ('tokens { money { integer_part: "20" currency: "xts" } }',),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_rejects_invalid(self, tokens):
        with pytest.raises(rewrite.Error):
            rewrite.top_rewrite(tokens, VERBALIZER)
