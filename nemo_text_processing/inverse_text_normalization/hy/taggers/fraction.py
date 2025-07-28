# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import INPUT_LOWER_CASED, GraphFst, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. երկու երրորդ -> tokens { fraction { numerator: "2" denominator: "3" } }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        (input_case is not necessary everything is made for lower_cased input)
        TODO add cased input support
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="fraction", kind="classify")
        cardinal_graph = cardinal.graph_no_exception
        quarter = pynini.string_map([("քառորդ", "4")])
        ordinal_graph = ordinal.graph | quarter

        numerator = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\"")
        denominator = pynutil.insert(" denominator: \"") + ordinal_graph + pynutil.insert("\"")

        final_graph = numerator + delete_space + denominator
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
