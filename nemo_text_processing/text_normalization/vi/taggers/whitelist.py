# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist for Vietnamese, e.g.
        "h" -> tokens { name: "giờ" }
        "p" -> tokens { name: "phút" }
        "s" -> tokens { name: "giây" }
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

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        # Add time units from time_units.tsv for better time handling
        if not deterministic:
            time_units_graph = _get_whitelist_graph(input_case, file=get_abs_path("data/time/time_units.tsv"))
            graph |= time_units_graph

        self.graph = graph
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.final_graph + pynutil.insert("\"")).optimize()

        # Add tokens wrapper
        self.fst = self.add_tokens(self.fst)
