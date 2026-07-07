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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space, load_labels


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        κ.λπ. -> tokens { name: "και τα λοιπά" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are
    defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")

        def _get_whitelist_graph(file):
            whitelist = load_labels(file)
            whitelist = [[x, y] for x, y in whitelist]
            return pynini.string_map(whitelist)

        whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv"))
        if input_file:
            whitelist |= _get_whitelist_graph(input_file)

        graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
        self.fst = graph.optimize()
