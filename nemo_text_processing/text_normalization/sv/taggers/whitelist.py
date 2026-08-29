# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst, convert_space
from nemo_text_processing.text_normalization.sv.graph_utils import TO_LOWER
from nemo_text_processing.text_normalization.sv.utils import get_abs_path, load_labels


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "s:t" -> tokens { name: "sankt" }
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

        def _get_case_insensitive_graph(input_case, file):
            graph = _get_whitelist_graph("lower_cased", file)
            if input_case == "lower_cased":
                return graph
            lower_cased_input = pynini.cdrewrite(TO_LOWER, "", "", NEMO_SIGMA)
            return lower_cased_input @ graph

        def _get_case_insensitive_optional_dot_graph(input_case, file):
            graph = _get_case_insensitive_graph(input_case, file)
            return graph + pynini.closure(pynutil.delete("."), 0, 1)

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist.tsv"))
        graph |= _get_case_insensitive_graph(input_case, get_abs_path("data/abbreviations/abbreviations.tsv"))
        graph |= _get_case_insensitive_optional_dot_graph(
            input_case, get_abs_path("data/abbreviations/abbreviations_opt_dot.tsv")
        )
        if not deterministic and input_case != "lower_cased":
            lower_cased_graph = _get_whitelist_graph("lower_cased", get_abs_path("data/whitelist.tsv"))
            graph |= pynutil.add_weight(lower_cased_graph, weight=0.0001)

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        if not deterministic:
            units_graph = _get_whitelist_graph(input_case, file=get_abs_path("data/measure/unit.tsv"))
            units_graph |= _get_whitelist_graph(input_case, file=get_abs_path("data/measure/unit_neuter.tsv"))
            units_graph |= _get_whitelist_graph(
                input_case, file=get_abs_path("data/abbreviations/nondeterministic.tsv")
            )
            units_graph |= _get_case_insensitive_graph(
                input_case, get_abs_path("data/abbreviations/abbreviations_alternatives.tsv")
            )
            graph |= units_graph

        self.graph = graph
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.final_graph + pynutil.insert("\"")).optimize()
