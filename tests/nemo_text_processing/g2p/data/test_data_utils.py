# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.g2p.data.data_utils import english_word_tokenize, ipa_word_tokenize


class TestDataUtils:
    @staticmethod
    def _create_expected_output(words):
        return [(word, False) for word in words]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize(self):
        input_text = "apple banana pear"
        expected_output = self._create_expected_output(["apple", " ", "banana", " ", "pear"])

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_punctuation(self):
        input_text = "Hello, world!"
        expected_output = self._create_expected_output(["hello", ", ", "world", "!"])

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_contractions(self):
        input_text = "It's a c'ntr'ction."
        expected_output = self._create_expected_output(["it's", " ", "a", " ", "c'ntr'ction", "."])

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_compound_words(self):
        input_text = "Forty-two is no run-off-the-mill number."
        expected_output = self._create_expected_output(
            ["forty-two", " ", "is", " ", "no", " ", "run-off-the-mill", " ", "number", "."]
        )

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_escaped(self):
        input_text = "Leave |this part UNCHANGED|."
        expected_output = [("leave", False), (" ", False), (["this", "part", "UNCHANGED"], True), (".", False)]

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_word_tokenize(self):
        input_text = "apple banana pear"
        expected_output = self._create_expected_output(["apple", " ", "banana", " ", "pear"])

        output = ipa_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_word_tokenize_with_accents(self):
        input_text = "The naïve piñata at the café..."
        expected_output = self._create_expected_output(
            ["the", " ", "naïve", " ", "piñata", " ", "at", " ", "the", " ", "café", "..."]
        )

        output = ipa_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_word_tokenize_with_numbers(self):
        input_text = "The 3D movie on 4-1-2022"
        expected_output = self._create_expected_output(["the", " ", "3d", " ", "movie", " ", "on", " ", "4-1-2022"])

        output = ipa_word_tokenize(input_text)
        assert output == expected_output
