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
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.pl.inflection import (
    load_adjective_abbreviations,
    load_ambiguous_abbreviations,
    load_inflected_abbreviations,
)
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels
from pynini.lib import pynutil


def _get_whitelist_graph(input_case: str, filepath: str) -> 'pynini.FstLike':
    labels = load_labels(filepath)
    if input_case == "lower_cased":
        labels = [[entry[0].lower()] + entry[1:] for entry in labels]
    return pynini.string_map(labels).optimize()


class WhiteListFst(GraphFst):
    """Classifies fixed and productively inflected Polish abbreviations."""

    def __init__(
        self, input_case: str, deterministic: bool = True, input_file: str = None
    ):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist.tsv"))
        if not deterministic and input_case != "lower_cased":
            graph |= pynutil.add_weight(
                _get_whitelist_graph("lower_cased", get_abs_path("data/whitelist.tsv")), 0.0001
            )

        self.inflected_graphs = load_inflected_abbreviations("data/abbreviations.tsv")
        graph |= pynini.union(*self.inflected_graphs.values())

        self.nondeterministic_graphs = load_ambiguous_abbreviations(
            "data/abbreviations_nondet.tsv"
        )
        self.adjective_graphs = load_adjective_abbreviations(
            "data/abbreviations_adjective_nondet.tsv"
        )
        if not deterministic:
            graph |= pynini.union(
                *self.nondeterministic_graphs.values(), *self.adjective_graphs.values()
            )

        if input_file:
            provided = _get_whitelist_graph(input_case, input_file)
            graph = graph | provided if not deterministic else provided

        self.graph = graph.optimize()
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert('name: "') + self.final_graph + pynutil.insert('"')).optimize()
