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
    Finite state transducer for classifying cardinals
        e.g. 23 -> cardinal { integer: "இருபத்திமூன்று" }
        "9999999999999999" -> cardinal { integer: "தொண்ணூற்றிஒன்பது கோடியே தொண்ணூற்றிஒன்பது லட்சத்து 
        தொண்ணூற்றிஒன்பதுஆயிரத்து ஒன்பதுநூற்று தொண்ணூற்றிஒன்பது கோடியே தொண்ணூற்றிஒன்பது லட்சத்து 
        தொண்ணூற்றிஒன்பதுஆயிரத்து ஒன்பதுநூற்று தொண்ணூற்றிஒன்பது" }

    Covers up to 16 digits (max 9999999999999999, just under 10^16),
    via composed கோடி (crore) groups.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """
    
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        def _extract_word(fst, name):
            return next(iter(pynini.compose(pynini.accep(name), fst).paths().ostrings()))

        scale = pynini.string_file(get_abs_path("data/numbers/scale.tsv"))

        special_units = pynini.string_file(get_abs_path("data/numbers/special_units.tsv"))
        special_units_input = pynini.project(special_units, "input")
        digit_oru = (
            special_units | pynini.compose(pynini.difference(NEMO_ALL_DIGIT, special_units_input), digit)
        ).optimize()

        # TEENS_AND_TIES (10-99)
        teens_and_ties_literal = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        tens_connector_stem = pynini.string_file(get_abs_path("data/numbers/tens_stem.tsv"))
        digit_3_to_9 = pynini.compose(pynini.difference(NEMO_ALL_DIGIT, special_units_input), digit).optimize()
        teens_and_ties_compositional = tens_connector_stem + digit_3_to_9

        teens_and_ties = pynini.union(teens_and_ties_literal, teens_and_ties_compositional).optimize()

        # HUNDREDS:
        hundred_stem = pynini.string_file(get_abs_path("data/numbers/hundred_stem.tsv"))
        hundred_suf_e = _extract_word(scale, "hundred_suf_e")
        hundred_suf_p = _extract_word(scale, "hundred_suf_p")
        hundred_exact = hundred_stem + pynutil.delete(NEMO_ALL_ZERO) ** 2 + pynutil.insert(hundred_suf_e)
        hundred_prefix = (hundred_stem + pynutil.insert(hundred_suf_p)).optimize()

        # ஆயிரம் (exact) and ஆயிரத்து (combining) share the same stem
        thousand_stem = pynini.string_file(get_abs_path("data/numbers/thousand.tsv"))
        thousand_suf_e = _extract_word(scale, "thousand_suf_e")
        thousand_suf_p = _extract_word(scale, "thousand_suf_p")
        thousand_exact = thousand_stem + pynutil.insert(thousand_suf_e)
        thousand_prefix = thousand_stem + pynutil.insert(thousand_suf_p)

        single_digit = digit | zero
        self.single_digits_graph = single_digit + pynini.closure(insert_space + single_digit)
        zero_del = pynutil.delete(NEMO_ALL_ZERO)

        def zdel(k):
            return zero_del**k if k > 0 else pynini.accep("")

        def scale_fn(head_exact, head_tail, n, tails):
            graph = head_exact
            for i, sub in enumerate(tails):
                graph |= head_tail + zdel(n - 1 - i) + insert_space + sub
            return graph.optimize()

        def band(base, exact_word, tail_word, n, tails):
            return scale_fn(base + pynutil.insert(exact_word) + zdel(n), base + pynutil.insert(tail_word), n, tails)

        # HUNDREDS (100-999): நூறு / நூற்று forms.
        graph_hundreds = scale_fn(hundred_exact, hundred_prefix, 2, [digit, teens_and_ties])
        self.graph_hundreds = graph_hundreds

        # THOUSANDS (1000-9999): ஆயிரம் / ஆயிரத்து forms
        graph_thousands = scale_fn(
            thousand_exact + zdel(3), thousand_prefix, 3, [digit, teens_and_ties, graph_hundreds]
        )
        self.graph_thousands = graph_thousands
        tails = [digit, teens_and_ties, graph_hundreds, graph_thousands]

        thousand_word = _extract_word(scale, "thousand_word_e")
        thousand_prefix_word = _extract_word(scale, "thousand_word_p")
        lakh_word = " " + _extract_word(scale, "lakh_word_e")
        lakh_prefix_word = " " + _extract_word(scale, "lakh_word_p")
        crore_word = " " + _extract_word(scale, "crore_word_e")
        crore_prefix_word = " " + _extract_word(scale, "crore_word_p")

        def add_scale(base, exact_word, prefix_word, n, tail_slice):
            """band() + append-to-tails for the common single-branch case."""
            g = band(base, exact_word, prefix_word, n, tails[:tail_slice])
            tails.append(g)
            return g

        # TEN-THOUSANDS (10^4): stem + ஆயிரம்
        graph_ten_thousands = add_scale(teens_and_ties, thousand_word, thousand_prefix_word, 3, 3)
        self.graph_ten_thousands = graph_ten_thousands

        # LAKHS / TEN-LAKHS (10^5, 10^6): stem + லட்சம்
        graph_lakhs = band(digit_oru, lakh_word, lakh_prefix_word, 5, tails[:5])
        self.graph_lakhs = graph_lakhs
        graph_ten_lakhs = band(teens_and_ties, lakh_word, lakh_prefix_word, 5, tails[:5])
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
        crore_graphs = [band(b, crore_word, crore_prefix_word, 7, tails) for b in crore_bases]
        graph_crores, graph_ten_crores = crore_graphs[0], crore_graphs[1]
        crore_graphs += [
            band(graph_crores, crore_word, crore_prefix_word, 7, tails),  # crores of crores
            band(graph_ten_crores, crore_word, crore_prefix_word, 7, tails),  # ten-crores of crores
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

        cardinal_with_leading_zeros = pynini.compose(
            NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph
        )

        delete_comma = pynutil.delete(",")
        digit3, digit2 = NEMO_ALL_DIGIT**3, NEMO_ALL_DIGIT**2

        western_format = pynini.closure(NEMO_ALL_DIGIT, 1, 3) + pynini.closure(delete_comma + digit3, 1)
        indian_format = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 2) + pynini.closure(delete_comma + digit2) + delete_comma + digit3
        )
        comma_number = western_format | indian_format
        cardinal_with_commas = pynini.compose(comma_number, graph_without_leading_zeros)

        self.final_graph = (
            graph_without_leading_zeros | cardinal_with_leading_zeros | cardinal_with_commas
        ).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
