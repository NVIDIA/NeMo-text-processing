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

import pynini
from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path, num_to_word
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g.

    Args:
        input_case:
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_digits = pynini.string_file(get_abs_path("data/numbers/digits.tsv"))
        graph_numbers = pynini.string_file(get_abs_path("data/numbers/tens.tsv"))
        graph_thousands = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

        final_graph = graph_digits | graph_numbers | graph_thousands
        final_graph = pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph

