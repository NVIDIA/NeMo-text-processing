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

from nemo_text_processing.text_normalization.kn.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_ALL_ZERO,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
       -23 -> cardinal { negative: "true"  integer: "ಇಪ್ಪತ್ತಮೂರು" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # base tables
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties = pynini.union(
            pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")),
            pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv")),
        )
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)
        hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties

        # single-digit reading (used for leading-zero numbers)
        single_digit = digit | zero
        self.single_digits_graph = single_digit + pynini.closure(insert_space + single_digit)

        delete_zero = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO), -0.1)
        EMPTY = pynini.accep("")  # epsilon, used when no zeros are deleted

        # suffixes (each defined once)
        suf_thousand = pynutil.insert(" ಸಾವಿರ")
        suf_thousand_gen = pynutil.insert(" ಸಾವಿರದ")
        suf_lakh = pynutil.insert(" ಲಕ್ಷ")
        suf_lakh_gen = pynutil.insert(" ಲಕ್ಷದ")
        suf_crore = pynutil.insert(" ಕೋಟಿ")
        suf_crore_gen = pynutil.insert(" ಕೋಟಿಯ")
        suf_thousand_crore = pynutil.insert(" ಸಾವಿರ ಕೋಟಿ")
        suf_thousand_crore_gen = pynutil.insert(" ಸಾವಿರ ಕೋಟಿಯ")
        suf_lakh_crore = pynutil.insert(" ಲಕ್ಷ ಕೋಟಿ")
        suf_lakh_crore_gen = pynutil.insert(" ಲಕ್ಷ ಕೋಟಿಯ")

        # generic scale builder
        # remainders: list of (graph, width) smallest -> largest, width = #digits it spans.
        # Returns (standalone_branch, remainder_union) so callers can reuse either piece.
        def scale_parts(coeff, num_zeros, suf_standalone, suf_gen, remainders):
            standalone = coeff + (delete_zero ** num_zeros) + suf_standalone
            with_rem = None
            for rem_graph, width in remainders:
                zeros = num_zeros - width
                middle = (delete_zero ** zeros) if zeros > 0 else EMPTY
                branch = coeff + middle + suf_gen + insert_space + rem_graph
                with_rem = branch if with_rem is None else with_rem | branch
            return standalone, with_rem

        def scale(coeff, num_zeros, suf_standalone, suf_gen, remainders):
            standalone, with_rem = scale_parts(coeff, num_zeros, suf_standalone, suf_gen, remainders)
            return (standalone | with_rem).optimize()

        # hundreds (special: standalone appends ು, no scale word)
        graph_hundreds = (
            hundreds + (delete_zero ** 2) + pynutil.insert("ು")   # 500 -> ಐನೂರು
            | hundreds + delete_zero + insert_space + digit         # 501 -> ಐನೂರ ಒಂದು
            | hundreds + insert_space + teens_ties                  # 523 -> ಐನೂರ ...
        ).optimize()
        self.graph_hundreds = graph_hundreds

        # remainder lists (each extends the previous one)
        rem_thousand = [(digit, 1), (teens_ties, 2), (graph_hundreds, 3)]

        # thousands / ten-thousands
        th_standalone, th_rem = scale_parts(digit, 3, suf_thousand, suf_thousand_gen, rem_thousand)
        tth_standalone, tth_rem = scale_parts(teens_and_ties, 3, suf_thousand, suf_thousand_gen, rem_thousand)
        graph_thousands = (th_standalone | th_rem).optimize()
        graph_ten_thousands = (tth_standalone | tth_rem).optimize()
        self.graph_thousands = graph_thousands
        self.graph_ten_thousands = graph_ten_thousands

        rem_lakh = rem_thousand + [(graph_thousands, 4), (graph_ten_thousands, 5)]

        # lakhs / ten-lakhs
        l_standalone, l_rem = scale_parts(digit, 5, suf_lakh, suf_lakh_gen, rem_lakh)
        tl_standalone, tl_rem = scale_parts(teens_and_ties, 5, suf_lakh, suf_lakh_gen, rem_lakh)
        graph_lakhs = (l_standalone | l_rem).optimize()
        graph_ten_lakhs = (tl_standalone | tl_rem).optimize()
        self.graph_lakhs = graph_lakhs
        self.graph_ten_lakhs = graph_ten_lakhs

        rem_crore = rem_lakh + [(graph_lakhs, 6), (graph_ten_lakhs, 7)]

        # any sub-crore remainder 1..99,99,999, read as a 7-digit field (handles leading zeros)
        sub_crore_rem = (
            (delete_zero ** 6) + digit
            | (delete_zero ** 5) + teens_and_ties
            | (delete_zero ** 4) + graph_hundreds
            | (delete_zero ** 3) + graph_thousands
            | (delete_zero ** 2) + graph_ten_thousands
            | delete_zero + graph_lakhs
            | graph_ten_lakhs
        ).optimize()

        # crores / ten-crores / hundred-crores
        graph_crores = scale(digit, 7, suf_crore, suf_crore_gen, rem_crore)
        graph_ten_crores = scale(teens_and_ties, 7, suf_crore, suf_crore_gen, rem_crore)
        graph_hundred_crores = scale(graph_hundreds, 7, suf_crore, suf_crore_gen, rem_crore)
        self.graph_crores = graph_crores
        self.graph_ten_crores = graph_ten_crores

        # big crore scales (10^10 .. 10^13)
        # part (b): reuse the remainder-variant coefficient graphs, then append ಕೋಟಿ.
        # `with_sub_rem` adds a full sub-crore remainder branch (1..99,99,999) after ಕೋಟಿಯ.
        def append_crore(coeff_remainder, with_sub_rem):
            g = coeff_remainder + (delete_zero ** 7) + suf_crore
            if with_sub_rem:
                g |= coeff_remainder + suf_crore_gen + insert_space + sub_crore_rem
            return g

        # thousand crores (10^10)
        part_a = scale(digit, 10, suf_thousand_crore, suf_thousand_crore_gen, rem_crore)
        graph_thousand_crores = (part_a | append_crore(th_rem, with_sub_rem=True)).optimize()

        # ten-thousand crores (10^11)
        part_a = scale(teens_and_ties, 10, suf_thousand_crore, suf_thousand_crore_gen, rem_crore)
        graph_ten_thousand_crores = (part_a | append_crore(tth_rem, with_sub_rem=True)).optimize()

        # lakh crores (10^12)
        part_a = scale(digit, 12, suf_lakh_crore, suf_lakh_crore_gen, rem_crore)
        graph_lakh_crores = (part_a | append_crore(l_rem, with_sub_rem=True)).optimize()

        # ten-lakh crores (10^13)
        part_a = scale(teens_and_ties, 12, suf_lakh_crore, suf_lakh_crore_gen, rem_crore)
        graph_ten_lakh_crores = (part_a | append_crore(tl_rem, with_sub_rem=True)).optimize()

        #   final composition (unchanged)
        graph_without_leading_zeros = pynini.union(
            digit,
            zero,
            teens_and_ties,
            graph_hundreds,
            graph_thousands,
            graph_ten_thousands,
            graph_lakhs,
            graph_ten_lakhs,
            graph_crores,
            graph_ten_crores,
            graph_hundred_crores,
            graph_thousand_crores,
            graph_ten_thousand_crores,
            graph_lakh_crores,
            graph_ten_lakh_crores,
        )
        self.graph_without_leading_zeros = graph_without_leading_zeros.optimize()

        # leading zeros -> digit-by-digit
        cardinal_with_leading_zeros = pynutil.add_weight(
            pynini.compose(NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph),
            0.5,
        )
        graph_no_commas = graph_without_leading_zeros | cardinal_with_leading_zeros

        # comma-separated numbers
        delete_comma = pynutil.delete(",")

        def exactly_n_digits(n):
            return pynini.closure(NEMO_ALL_DIGIT, n, n)

        western_format = pynini.closure(NEMO_ALL_DIGIT, 1, 3) + pynini.closure(delete_comma + exactly_n_digits(3), 1)
        indian_format = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 2)
            + pynini.closure(delete_comma + exactly_n_digits(2))
            + delete_comma
            + exactly_n_digits(3)
        )
        comma_number = western_format | indian_format
        cardinal_with_commas = pynutil.add_weight(
            pynini.compose(comma_number, graph_without_leading_zeros), 0.1
        )

        final_graph = graph_no_commas | cardinal_with_commas
        self.final_graph = final_graph.optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph)