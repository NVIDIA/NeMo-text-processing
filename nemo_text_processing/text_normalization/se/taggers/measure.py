# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023, 2026, Jim O'Regan for Språkbanken Tal
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_zero_or_one_space
from nemo_text_processing.text_normalization.se.utils import get_abs_path


class MeasureFst(GraphFst):
    """Classifies documented Northern Sámi cardinal measure expressions."""

    def __init__(self, cardinal: GraphFst, decimal=None, fraction=None, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        unit_nominative = pynini.string_file(get_abs_path("data/measure/unit_simple.tsv"))
        unit_genitive = pynini.string_file(get_abs_path("data/measure/unit_genitive.tsv"))
        unit_rate = pynini.string_file(get_abs_path("data/measure/unit_rate.tsv"))

        one = pynini.accep("1") @ cardinal.graph
        non_one = pynini.difference(pynini.project(cardinal.graph, "input"), "1") @ cardinal.graph

        def cardinal_token(graph):
            return pynutil.insert('cardinal { integer: "') + graph + pynutil.insert('" } ')

        def unit_token(graph):
            return pynutil.insert('units: "') + graph + pynutil.insert('"')

        separator = delete_zero_or_one_space
        singular = cardinal_token(one) + separator + unit_token(unit_nominative)
        governed = cardinal_token(non_one) + separator + unit_token(unit_genitive)
        rate = cardinal_token(cardinal.graph) + separator + unit_token(unit_rate)

        # Riektačállinrávvagat (Sámediggi, revised 2019), p. 57:
        # "ovcce- ja guoktenuppelohjahkáččat" corresponds to
        # "9- ja 12-jahkásaččat".
        age_adjective = pynini.cross("jahkásaš", "jahkásaš") | pynini.cross("jahkásaččat", "jahkáččat")
        if not deterministic:
            age_adjective |= pynini.cross("jahkásaččat", "jahkásaččat")
        age = cardinal.compound + pynutil.delete("-") + age_adjective
        age_token = pynutil.insert('name: "') + age + pynutil.insert('"')

        if not deterministic:
            governed |= cardinal_token(non_one) + separator + unit_token(unit_nominative)

        self.fst = (self.add_tokens(singular | governed | rate) | age_token).optimize()
        self.graph = (
            one + separator + pynutil.insert(" ") + unit_nominative
            | non_one + separator + pynutil.insert(" ") + unit_genitive
            | cardinal.graph + separator + pynutil.insert(" ") + unit_rate
            | age
        ).optimize()
