# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst, string_map_cased
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import convert_space, insert_space


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. misses -> tokens { name: "mrs." }
    This class has highest priority among all classifier grammars.
    Whitelisted tokens are defined and loaded from "data/whitelist.tsv" (unless input_file specified).

    Args:
        input_file: path to a file with whitelist replacements (each line of the file: written_form\tspoken_form\n),
            e.g. nemo_text_processing/inverse_text_normalization/he/data/whitelist.tsv
    """

    def __init__(self, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")
        prefix_graph = pynini.string_file(get_abs_path("data/prefix.tsv"))

        if input_file is None:
            input_file = get_abs_path("data/whitelist.tsv")

        if not os.path.exists(input_file):
            raise ValueError(f"Whitelist file {input_file} not found")

        optional_prefix_graph = pynini.closure(
            pynutil.insert('morphosyntactic_features: "') + prefix_graph + pynutil.insert('"') + insert_space,
            0,
            1,
        )
        whitelist = string_map_cased(input_file)
        graph = pynutil.insert('name: "') + convert_space(whitelist) + pynutil.insert('"')
        final_graph = optional_prefix_graph + graph
        self.fst = final_graph.optimize()
