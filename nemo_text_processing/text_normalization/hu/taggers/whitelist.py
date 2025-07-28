# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.hu.utils import get_abs_path, load_labels, naive_inflector


def load_inflected(filename, input_case, singular_only=False, skip_spaces=True):
    forms = []
    with open(filename) as tsv:
        for line in tsv.readlines():
            parts = line.strip().split("\t")
            key = parts[0]
            if input_case == "lower_cased":
                key = parts[0].lower()
            forms.append((key, parts[1]))
            if not (skip_spaces and " " in parts[1]):
                forms += naive_inflector(key, parts[1], singular_only)
    graph = pynini.string_map(forms)
    return graph


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "stb." -> tokens { name: "s a többi" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(input_case, file):
            whitelist = load_labels(file)
            if input_case == "lower_cased":
                whitelist = [[x[0].lower()] + x[1:] for x in whitelist]
            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist.tsv"))
        if not deterministic and input_case != "lower_cased":
            graph |= pynutil.add_weight(
                _get_whitelist_graph("lower_cased", get_abs_path("data/whitelist.tsv")), weight=0.0001
            )

        graph_inflected = load_inflected(get_abs_path("data/whitelist_inflect.tsv"), input_case, False)
        graph_inflected_sg = load_inflected(get_abs_path("data/whitelist_inflect_sg.tsv"), input_case, True)
        units_graph = load_inflected(get_abs_path("data/measures/measurements.tsv"), input_case, False)

        graph |= graph_inflected
        graph |= graph_inflected_sg

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        if not deterministic:
            graph |= units_graph

        self.graph = graph
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.final_graph + pynutil.insert("\"")).optimize()
