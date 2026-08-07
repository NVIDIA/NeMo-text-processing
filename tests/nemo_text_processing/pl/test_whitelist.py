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
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from nemo_text_processing.text_normalization.pl.taggers.whitelist import WhiteListFst

from ..utils import parse_test_case_file


class TestWhitelist:
    normalizer = Normalizer(input_case="cased", lang="pl", cache_dir=None, post_process=False)
    audio_normalizer = NormalizerWithAudio(
        input_case="cased", lang="pl", cache_dir=None, post_process=False
    )

    @parameterized.expand(parse_test_case_file("pl/data_text_normalization/test_cases_whitelist.txt"))
    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        prediction = self.normalizer.normalize(test_input, punct_post_process=False)
        assert prediction == expected

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_inflected_graphs_are_keyed_by_slot(self):
        whitelist = WhiteListFst(input_case="cased", deterministic=True)
        assert rewrite.one_top_rewrite("s-ka", whitelist.inflected_graphs["sg_nom"]) == "spółka"
        assert rewrite.one_top_rewrite("s-ki", whitelist.inflected_graphs["sg_gen"]) == "spółki"
        assert rewrite.one_top_rewrite("s-ce", whitelist.inflected_graphs["sg_loc"]) == "spółce"

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_ambiguous_entry_has_all_singular_forms(self):
        whitelist = WhiteListFst(input_case="cased", deterministic=False)
        expected = {
            "sg_nom": "rok",
            "sg_gen": "roku",
            "sg_dat": "rokowi",
            "sg_acc": "rok",
            "sg_ins": "rokiem",
            "sg_loc": "roku",
            "sg_voc": "roku",
        }
        for slot, spoken in expected.items():
            assert rewrite.one_top_rewrite("r.", whitelist.nondeterministic_graphs[slot]) == spoken
