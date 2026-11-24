# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane AI Labs. All rights reserved.
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

"""What a case file cannot check for Luganda: grammar caching, cased input, the command line and the export verbalizer."""

import sys

import pytest
from parameterized import parameterized
from pynini.lib import rewrite

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer, parse_args
from nemo_text_processing.inverse_text_normalization.lg.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.inverse_text_normalization.lg.verbalizers.verbalize import VerbalizeFst

from ..utils import CACHE_DIR

SAMPLES = ("amagi abiri", "ssiringi za uganda abiri", "abantu emitwalo abiri", "ente bbiri mu nnimiro")


class TestLugandaGrammar:

    inverse_normalizer = InverseNormalizer(lang='lg', cache_dir=CACHE_DIR, overwrite_cache=False)

    # The compiled classifier is saved to and restored from a FAR file with the same results.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_far_cache_roundtrip(self, tmp_path):
        ClassifyFst(cache_dir=str(tmp_path), overwrite_cache=True)
        assert (tmp_path / "lg_itn_lower_cased.far").exists()
        restored = InverseNormalizer(lang='lg', cache_dir=str(tmp_path))
        for spoken in SAMPLES:
            expected = self.inverse_normalizer.inverse_normalize(spoken, verbose=False)
            assert restored.inverse_normalize(spoken, verbose=False) == expected

    # Cased input builds the same grammar: words with capitals pass through unchanged.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_cased_input_case(self, tmp_path):
        cased = InverseNormalizer(lang='lg', cache_dir=str(tmp_path), input_case='cased')
        assert (tmp_path / "lg_itn_cased.far").exists()
        for spoken in SAMPLES:
            expected = self.inverse_normalizer.inverse_normalize(spoken, verbose=False)
            assert cased.inverse_normalize(spoken, verbose=False) == expected

    # The n' split sees the words with spaces already collapsed, as the kernel does.
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_split_after_repeated_spaces(self):
        pred = self.inverse_normalizer.inverse_normalize("yampa emmere n'emitwalo  \t ebiri", verbose=False)
        assert pred == "yampa emmere ne 20,000 UGX"

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_offered_on_cli(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["inverse_normalize.py", "--language", "lg", "--text", "amagi abiri"])
        assert parse_args().language == "lg"


class TestLugandaExportVerbalizer:

    verbalizer = VerbalizeFst().fst

    # Used by the grammar export (Sparrowhawk): formatting only, values come resolved from the classifier.
    @parameterized.expand(
        [
            ('cardinal { integer: "2" }', "2"),
            ('cardinal { integer: "999999999999" }', "999,999,999,999"),
            ('money { integer_part: "20000" currency: "UGX" }', "20,000 UGX"),
        ]
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_formatting(self, tokens, expected):
        assert rewrite.top_rewrite(tokens, self.verbalizer) == expected
