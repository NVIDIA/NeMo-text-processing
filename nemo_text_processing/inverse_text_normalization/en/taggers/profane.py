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
from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    capitalized_input_graph,
)
from pynini.lib import pynutil


class ProfaneFst(GraphFst):
    """
    Finite state transducer for classifying profane words
        e.g. bitch -> profane { filtered: "b****" }

    This class has highest priority among all classifier grammars

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        input_file: path to a file with profane words to be redacted with "*" symbol. (each line of the file: spoken_form\n)
            e.g. nemo_text_processing/inverse_text_normalization/en/data/swear_sequences.tsv
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED, input_file: str = None):
        super().__init__(name="profane", kind="classify")
        # Profane Grammar
        if input_file is None:
            input_file = get_abs_path("data/swear_sequences.tsv")

        if not os.path.exists(input_file):
            raise ValueError(f"Profane words file {input_file} not found")

        profane_graph = pynini.string_file(get_abs_path("data/swear_sequences.tsv"))

        bowdlerize = (
            (NEMO_ALPHA | NEMO_DIGIT) + pynini.closure(pynini.cross(NEMO_SPACE | NEMO_ALPHA | NEMO_DIGIT, "*"), 1)
        ).optimize()

        profane_graph = (profane_graph @ bowdlerize).optimize()

        if input_case == INPUT_CASED:
            profane_graph = capitalized_input_graph(profane_graph)

        # Token insertion
        final_profane_graph = pynutil.insert('filtered: "') + profane_graph + pynutil.insert('"')

        # Inserts the profane tag
        final_profane_graph = self.add_tokens(final_profane_graph)
        self.fst = final_profane_graph.optimize()
