# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal numbers, e.g.
        zehnter -> ordinal { integer: "10" }
        dreiundzwanzigstes -> ordinal { integer: "23" }

    Single-digit ordinals are not tagged, so they stay spelled out, e.g.
        dritter -> dritter
    They are still exposed as digits in self.graph_ordinals, e.g.
        dritter -> 3.
    for other semiotic classes such as date.

    Args:
        cardinal: ITN Cardinal Tagger
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        # the cardinal grammar groups digits with a period, an ordinal is written without it
        remove_separators = pynini.cdrewrite(pynutil.delete("."), "", "", NEMO_SIGMA)
        cardinal_graph = (cardinal.graph_no_exception @ remove_separators).optimize()

        # ordinal stems that differ from the cardinal, e.g. "drit" -> "drei", "zwanzigs" -> "zwanzig"
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/irregular_digits.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv"))
        graph_thousands = pynini.string_file(get_abs_path("data/ordinals/thousands.tsv"))
        ordinal_stem = graph_digit | graph_ties | graph_thousands

        suffixes = pynini.string_file(get_abs_path("data/ordinals/suffixes.tsv"))

        # only words that actually carry an ending, so plain cardinals are not tagged as ordinals
        ends_with_suffix = (NEMO_SIGMA + suffixes).optimize()

        # a stem outside the tables is already the cardinal, so the rewrite only drops the ending
        to_cardinal_words = pynini.cdrewrite(
            pynini.closure(ordinal_stem, 0, 1) + pynutil.delete(suffixes),
            "",
            "[EOS]",
            NEMO_SIGMA,
        )

        graph = (ends_with_suffix @ to_cardinal_words).optimize() @ cardinal_graph

        self.graph_ordinals = (graph + pynutil.insert(".")).optimize()

        at_least_two_digits = NEMO_DIGIT + pynini.closure(NEMO_DIGIT, 1)
        graph_standalone = graph @ at_least_two_digits

        final_graph = pynutil.insert('integer: "') + graph_standalone + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
