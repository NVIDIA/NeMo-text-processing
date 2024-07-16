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
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    GraphFst,
)
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties.tsv"))
        graph = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_digit, graph_teens, pynini.cross("वाँ", ""), pynini.cross("वां","")
        )

        graph_fem_digit = pynini.string_file(get_abs_path("data/ordinals/digit_fem.tsv"))
        graph_fem_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties_fem.tsv"))
        graph_fem = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_fem_digit, graph_fem_teens, pynini.cross("वी", "")
        )
        graph = pynini.compose(graph, cardinal_graph)
        graph_fem = pynini.compose(graph_fem, cardinal_graph)

        morpho_graph = pynutil.insert("morphosyntactic_features: \"वाँ\"")
        morpho_graph_fem = pynutil.insert("morphosyntactic_features: \"वी\"")

        final_graph = (pynutil.insert("integer: \"") + graph + pynutil.insert("\" ") + morpho_graph)
        final_graph |= (pynutil.insert("integer: \"") + graph_fem + pynutil.insert("\" ") + morpho_graph_fem)
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
