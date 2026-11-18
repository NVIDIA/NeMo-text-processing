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
import pynini
import pytest

from nemo_text_processing.text_normalization.pl.taggers.tokenize_and_classify_with_audio import ClassifyFst


class TestAudioNormalizer:
    normalizer = ClassifyFst(input_case="cased")

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_ngram_acceptor_selects_inflection(self):
        masculine = pynini.accep("Mam dwadzieścia dwa koty", weight=1)
        feminine = pynini.accep("Mam dwadzieścia dwie koty", weight=0)
        language_model = (masculine | feminine).optimize()
        assert self.normalizer.normalize("Mam 22 koty", language_model) == "Mam dwadzieścia dwie koty"

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_ngram_acceptor_selects_compound_tokenization(self):
        joined = pynini.accep("To dwudziestodwulatka", weight=1)
        spaced = pynini.accep("To dwudziesto dwu latka", weight=0)
        language_model = (joined | spaced).optimize()
        assert self.normalizer.normalize("To 22-latka", language_model) == "To dwudziesto dwu latka"
