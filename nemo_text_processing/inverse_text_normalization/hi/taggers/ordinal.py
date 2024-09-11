# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_CHAR,
    NEMO_HI_DIGIT,
    NEMO_SIGMA,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. तेरहवां -> ordinal { integer: "१३" }

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties.tsv"))
        graph_digit_hundred = pynini.string_file(get_abs_path("data/ordinals/hundred_digit.tsv"))
        graph = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_digit,
            graph_teens,
            graph_digit_hundred,
            pynini.cross("वाँ", "वाँ"),
            pynini.cross("वां", "वां"),
            pynini.cross("वें", "वें"),
            pynini.cross("वे", "वे"),
        )

        graph_fem_digit = pynini.string_file(get_abs_path("data/ordinals/digit_fem.tsv"))
        graph_fem_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties_fem.tsv"))
        graph_digit_hundred_fem = pynini.string_file(get_abs_path("data/ordinals/hundred_digit_fem.tsv"))
        graph_fem = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_fem_digit,
            graph_fem_teens,
            graph_digit_hundred_fem,
            pynini.cross("वीं", "वीं"),
            pynini.cross("वी", "वी"),
        )
        graph = pynini.compose(
            graph | graph_fem,
            (
                cardinal_graph
                + pynini.union(
                    pynini.cross("वाँ", "वाँ"),
                    pynini.cross("वां", "वां"),
                    pynini.cross("वीं", "वीं"),
                    pynini.cross("वी", "वी"),
                    pynini.cross("वें", "वें"),
                    pynini.cross("वे", "वे"),
                )
            ),
        ).optimize()

        morph_features_graph = pynini.string_file(get_abs_path("data/ordinals/morph_features.tsv"))
        morpho_graph = pynutil.insert("\" morphosyntactic_features: \"") + morph_features_graph + pynutil.insert("\"")

        rule = pynini.cdrewrite(morpho_graph, pynini.closure(NEMO_HI_DIGIT), pynini.union("[EOS]", " "), NEMO_SIGMA)

        final_graph = pynutil.insert("integer: \"") + graph @ rule
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
