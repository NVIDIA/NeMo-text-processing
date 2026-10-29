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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions, e.g.
        ein halb -> fraction { numerator: "1" denominator: "2" }
        ein ein halb -> fraction { integer_part: "1" numerator: "1" denominator: "2" }
        minus ein zwei und zwanzigstel -> fraction { negative: "-" numerator: "1" denominator: "22" }

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        # the cardinal matches the empty string, so a field could be filled without reading input
        non_empty = pynini.difference(NEMO_SIGMA, "").optimize()
        cardinal_graph = (non_empty @ cardinal.graph_no_exception).optimize()

        remove_separators = pynini.cdrewrite(pynutil.delete("."), "", "", NEMO_SIGMA)
        cardinal_no_separators = (cardinal_graph @ remove_separators).optimize()

        suffixes = pynini.string_file(get_abs_path("data/fractions/suffixes.tsv"))
        half = pynini.string_file(get_abs_path("data/fractions/half.tsv"))

        # a German denominator is the ordinal stem plus -tel
        to_cardinal_words = pynini.cdrewrite(
            pynini.closure(ordinal.ordinal_stem, 0, 1) + pynutil.delete(suffixes),
            "",
            "[EOS]",
            NEMO_SIGMA,
        )
        ends_with_suffix = (pynini.closure(NEMO_CHAR, 1) + suffixes).optimize()
        denominator_words = (ends_with_suffix @ to_cardinal_words) | half

        integer_part = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        numerator = pynutil.insert('numerator: "') + cardinal_no_separators + pynutil.insert('"')
        denominator = (
            pynutil.insert('denominator: "') + (denominator_words @ cardinal_no_separators) + pynutil.insert('"')
        )

        # the space between fields is read from the input, so ein dreizehntel cannot read as ein drei zehntel
        self.graph_wo_negative = (
            pynini.closure(integer_part + NEMO_SPACE, 0, 1) + numerator + NEMO_SPACE + denominator
        ).optimize()

        final_graph = cardinal.optional_minus_graph + self.graph_wo_negative

        # the graph the other German semiotic classes consume, without the first-dozen exception
        integer_chars = pynini.closure(NEMO_DIGIT | pynini.accep("."), 1)
        digits = pynini.closure(NEMO_DIGIT, 1)
        graph_fraction_raw = (
            pynini.closure(pynutil.delete('negative: "') + pynini.accep("-") + pynutil.delete('" '), 0, 1)
            + pynini.closure(pynutil.delete('integer_part: "') + integer_chars + pynini.cross('" ', " "), 0, 1)
            + pynutil.delete('numerator: "')
            + digits
            + pynini.cross('" ', "/")
            + pynutil.delete('denominator: "')
            + digits
            + pynutil.delete('"')
        )
        self.graph = (final_graph @ graph_fraction_raw).optimize()

        self.fst = self.add_tokens(final_graph).optimize()
