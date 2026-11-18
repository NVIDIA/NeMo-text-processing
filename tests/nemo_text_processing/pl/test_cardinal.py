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
from nemo_text_processing.text_normalization.pl.taggers.cardinal import CardinalFst

from ..utils import parse_test_case_file


class TestCardinal:
    normalizer = Normalizer(input_case="cased", lang="pl", cache_dir=None, post_process=False)

    @parameterized.expand(parse_test_case_file("pl/data_text_normalization/test_cases_cardinal.txt"))
    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        prediction = self.normalizer.normalize(test_input, punct_post_process=False)
        assert prediction == expected

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_inflectional_graphs(self):
        cardinal = CardinalFst()
        assert rewrite.one_top_rewrite("1", cardinal.graphs["f_sg_nom"]) == "jedna"
        assert rewrite.one_top_rewrite("2", cardinal.graphs["mp_pl_nom"]) == "dwaj"
        assert rewrite.one_top_rewrite("22", cardinal.graphs["mp_pl_nom"]) == "dwudziestu dwóch"
        assert rewrite.one_top_rewrite("22", cardinal.graphs["f_pl_nom"]) == "dwadzieścia dwie"
        assert rewrite.one_top_rewrite("22", cardinal.graphs["pl_gen"]) == "dwudziestu dwóch"

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_compound_graph(self):
        cardinal = CardinalFst()
        assert rewrite.one_top_rewrite("22", cardinal.graphs["compound"]) == "dwudziestodwu"
        assert rewrite.one_top_rewrite("22-latka", cardinal.compound) == "dwudziestodwulatka"

        cardinal = CardinalFst(deterministic=False)
        alternatives = rewrite.top_rewrites("22", cardinal.graphs["compound"], 10)
        assert "dwudziestodwu" in alternatives
        assert "dwudziesto dwu" in alternatives
        alternatives = rewrite.top_rewrites("22-latka", cardinal.compound, 20)
        assert "dwudziestodwulatka" in alternatives
        assert "dwudziesto dwu latka" in alternatives

    @pytest.mark.run_only_on("CPU")
    @pytest.mark.unit
    def test_noun_graphs(self):
        cardinal = CardinalFst(deterministic=False)
        assert rewrite.one_top_rewrite("2-ka", cardinal.noun_graphs["sg_nom"]) == "dwójka"
        assert rewrite.one_top_rewrite("2ką", cardinal.noun_graphs["sg_ins"]) == "dwójką"
        assert rewrite.one_top_rewrite("11-ce", cardinal.noun_graphs["sg_loc"]) == "jedenastce"
        assert rewrite.one_top_rewrite("20-ek", cardinal.noun_graphs["pl_gen"]) == "dwudziestek"
        assert rewrite.one_top_rewrite("200-kami", cardinal.noun_graphs["pl_ins"]) == "dwusetkami"
