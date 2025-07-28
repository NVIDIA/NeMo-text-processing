# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    convert_space,
    string_map_cased,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. misses -> tokens { name: "mrs." }
    This class has highest priority among all classifier grammars.
    Whitelisted tokens are defined and loaded from "data/whitelist.tsv" (unless input_file specified).

    Args:
        input_file: path to a file with whitelist replacements (each line of the file: written_form\tspoken_form\n),
            e.g. nemo_text_processing/inverse_text_normalization/en/data/whitelist.tsv
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")

        if input_file is None:
            input_file = get_abs_path("data/whitelist.tsv")

        if not os.path.exists(input_file):
            raise ValueError(f"Whitelist file {input_file} not found")

        whitelist = string_map_cased(input_file, input_case)
        graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
        self.fst = graph.optimize()
