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
from pynini.lib import rewrite

from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from nemo_text_processing.text_normalization.pl.taggers.abbreviation import AbbreviationFst


class TestAbbreviation:
    normalizer = NormalizerWithAudio(input_case="cased", lang="pl", cache_dir=None, post_process=False)

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_graph(self):
        abbreviation = AbbreviationFst(deterministic=False)
        assert 'abbreviation { value: "A B C" }' in rewrite.top_rewrites("ABC", abbreviation.fst, 10)
        assert 'abbreviation { value: "A. B. C." }' in rewrite.top_rewrites("A.B.C.", abbreviation.fst, 10)

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_audio_lattice(self):
        predictions = self.normalizer.normalize("Kod ABC", n_tagged=20, punct_post_process=False)
        assert "Kod A B C" in predictions
