# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ta.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_ALL_ZERO,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -23 -> cardinal { negative: "true"  integer: "இருபத்து மூன்று" }
    The highest unit used is கோடி.
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        def sf(name):
            return pynini.string_file(get_abs_path(f"data/numbers/{name}.tsv"))

        digit = sf("digit")
        zero = sf("zero")
        teens_ties = pynini.union(sf("teens_and_ties"), sf("teens_and_ties_en"))
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        # digit_oru == digit except "1" takes its prefixing form ஒரு (before scale units)
        one_oru = pynini.cross("1", "ஒரு") | pynini.cross("௧", "ஒரு")
        digit_oru = (
            one_oru | pynini.compose(pynini.difference(NEMO_ALL_DIGIT, pynini.union("1", "௧")), digit)
        ).optimize()

        # single hundreds table: absolute forms (நூறு) are keyed by "X00",
        # combining forms (நூற்று) by "X"; split them apart by input length.
        hundred = sf("hundred_ta")
        hundred_ta = pynini.compose(NEMO_ALL_DIGIT**3, hundred).optimize()
        hundred_prefix = pynini.compose(NEMO_ALL_DIGIT, hundred).optimize()

        # ஆயிரம் (exact) and ஆயிரத்து (combining) share the same stem
        thousand_stem = sf("thousand")
        thousand_exact = thousand_stem + pynutil.insert("ம்")
        thousand_prefix = thousand_stem + pynutil.insert("த்து")

        self.digit = digit
        self.zero = zero
        self.hundred_ta = hundred_ta
        self.teens_and_ties = teens_and_ties

        single_digit = digit | zero
        self.single_digits_graph = single_digit + pynini.closure(insert_space + single_digit)
        zero_del = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO), -0.1)

        def zdel(k):
            # NOTE: pynini treats ``fst ** 0`` as Kleene-star, so guard the 0 case.
            return zero_del**k if k > 0 else pynini.accep("")

        def scale(head_exact, head_tail, n, tails):
            """One magnitude band: exact multiple + every remainder combination.

            ``head_exact`` consumes the whole exact value; ``head_tail`` consumes only
            the leading part and is followed by the deleted trailing zeros, a space and
            a smaller sub-number (``tails`` are ordered smallest magnitude first).
            """
            graph = head_exact
            for i, sub in enumerate(tails):
                graph |= head_tail + zdel(n - 1 - i) + insert_space + sub
            return graph.optimize()

        def band(base, exact_word, tail_word, n, tails):
            """Magnitude band whose head is a number stem plus an inserted unit word."""
            return scale(base + pynutil.insert(exact_word) + zdel(n), base + pynutil.insert(tail_word), n, tails)

        # HUNDREDS (100-999): நூறு / நூற்று forms
        graph_hundreds = scale(pynutil.add_weight(hundred_ta, -1.0), hundred_prefix, 2, [single_digit, teens_ties])
        self.graph_hundreds = graph_hundreds

        # THOUSANDS (1000-9999): ஆயிரம் / ஆயிரத்து forms
        graph_thousands = scale(
            thousand_exact + zdel(3), thousand_prefix, 3, [single_digit, teens_ties, graph_hundreds]
        )
        self.graph_thousands = graph_thousands

        # ladder of remainder fillers, smallest magnitude first
        tails = [single_digit, teens_ties, graph_hundreds, graph_thousands]

        # TEN-THOUSANDS (10^4): stem + ஆயிரம்
        graph_ten_thousands = band(teens_and_ties, "ஆயிரம்", "ஆயிரத்து", 3, tails[:3])
        self.graph_ten_thousands = graph_ten_thousands
        tails.append(graph_ten_thousands)

        # LAKHS / TEN-LAKHS (10^5, 10^6): stem + லட்சம்
        graph_lakhs = band(digit_oru, " லட்சம்", " லட்சத்து", 5, tails[:5])
        self.graph_lakhs = graph_lakhs
        graph_ten_lakhs = band(teens_and_ties, " லட்சம்", " லட்சத்து", 5, tails[:5])
        self.graph_ten_lakhs = graph_ten_lakhs
        tails += [graph_lakhs, graph_ten_lakhs]

        # CRORES and higher (10^7 .. 10^15): stem + கோடி
        crore_bases = [
            digit_oru,  # crores
            teens_and_ties,  # ten-crores
            graph_hundreds,  # hundreds of crores
            graph_thousands,  # thousands of crores
            graph_ten_thousands,  # ten-thousands of crores
            graph_lakhs,  # lakhs of crores
            graph_ten_lakhs,  # ten-lakhs of crores
        ]
        crore_graphs = [band(b, " கோடி", " கோடியே", 7, tails) for b in crore_bases]
        graph_crores, graph_ten_crores = crore_graphs[0], crore_graphs[1]
        crore_graphs += [
            band(graph_crores, " கோடி", " கோடியே", 7, tails),  # crores of crores
            band(graph_ten_crores, " கோடி", " கோடியே", 7, tails),  # ten-crores of crores
        ]

        # FINAL GRAPH
        graph_without_leading_zeros = pynini.union(
            digit,
            zero,
            teens_and_ties,
            graph_hundreds,
            graph_thousands,
            graph_ten_thousands,
            graph_lakhs,
            graph_ten_lakhs,
            *crore_graphs,
        )
        self.graph_without_leading_zeros = graph_without_leading_zeros.optimize()

        cardinal_with_leading_zeros = pynutil.add_weight(
            pynini.compose(NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph), 0.5
        )
        self.final_graph = (self.graph_without_leading_zeros | cardinal_with_leading_zeros).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph)
