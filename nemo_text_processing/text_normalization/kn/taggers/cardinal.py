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
       23 -> cardinal { negative: "true"  integer: "ಇಪ್ಪತ್ತಮೂರು" }
 
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)


        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_and_ties = pynini.union(
            pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")),
            pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv")),
        )
        hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))

        single_digit = digit | zero
        self.single_digits_graph = single_digit + pynini.closure(insert_space + single_digit)

        delete_zero = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO), -0.1)
        EMPTY = pynini.accep("")
        
        scale_suffixes = pynini.string_file(get_abs_path("data/numbers/scale_suffixes.tsv"))

        def suffix_insert(key, leading_space=True):
            value = pynini.compose(key, scale_suffixes).string()
            text = (" " + value) if leading_space else value
            return pynutil.insert(text)

        suf_thousand = suffix_insert("thousand")
        suf_thousand_gen = suffix_insert("thousand_gen")
        suf_lakh = suffix_insert("lakh")
        suf_lakh_gen = suffix_insert("lakh_gen")
        suf_crore = suffix_insert("crore")
        suf_crore_gen = suffix_insert("crore_gen")
        suf_thousand_crore = suffix_insert("thousand_crore")
        suf_thousand_crore_gen = suffix_insert("thousand_crore_gen")
        suf_lakh_crore = suffix_insert("lakh_crore")
        suf_lakh_crore_gen = suffix_insert("lakh_crore_gen")
        suf_hundred = suffix_insert("hundred_suffix", leading_space=False)

        def scale_parts(coeff, num_zeros, suf_standalone, suf_gen, remainders):
            standalone = coeff + (delete_zero**num_zeros) + suf_standalone
            with_rem = None

            for rem_graph, width in remainders:
                zeros = num_zeros - width
                middle = (delete_zero**zeros) if zeros > 0 else EMPTY
                branch = coeff + middle + suf_gen + insert_space + rem_graph
                with_rem = branch if with_rem is None else with_rem | branch

            return standalone, with_rem

        def scale(coeff, num_zeros, suf_standalone, suf_gen, remainders):
            standalone, with_rem = scale_parts(coeff, num_zeros, suf_standalone, suf_gen, remainders)
            return (standalone | with_rem).optimize()
       
        graph_hundreds = (
            hundreds + (delete_zero**2) + suf_hundred  
            | hundreds + delete_zero + insert_space + digit 
            | hundreds + insert_space + teens_and_ties
        ).optimize()

        self.graph_hundreds = graph_hundreds

        rem_thousand = [(digit, 1), (teens_and_ties, 2), (graph_hundreds, 3)]

        th_standalone, th_rem = scale_parts(digit, 3, suf_thousand, suf_thousand_gen, rem_thousand)
        tth_standalone, tth_rem = scale_parts(teens_and_ties, 3, suf_thousand, suf_thousand_gen, rem_thousand)
        
        graph_thousands = (th_standalone | th_rem).optimize()
        graph_ten_thousands = (tth_standalone | tth_rem).optimize()

        self.graph_thousands = graph_thousands
        self.graph_ten_thousands = graph_ten_thousands

        rem_lakh = rem_thousand + [(graph_thousands, 4), (graph_ten_thousands, 5)]

        l_standalone, l_rem = scale_parts(digit, 5, suf_lakh, suf_lakh_gen, rem_lakh)
        tl_standalone, tl_rem = scale_parts(teens_and_ties, 5, suf_lakh, suf_lakh_gen, rem_lakh)
        graph_lakhs = (l_standalone | l_rem).optimize()
        graph_ten_lakhs = (tl_standalone | tl_rem).optimize()

        self.graph_lakhs = graph_lakhs
        self.graph_ten_lakhs = graph_ten_lakhs

        rem_crore = rem_lakh + [(graph_lakhs, 6), (graph_ten_lakhs, 7)]

        sub_crore_rem = (
            (delete_zero**6) + digit
            | (delete_zero**5) + teens_and_ties
            | (delete_zero**4) + graph_hundreds
            | (delete_zero**3) + graph_thousands
            | (delete_zero**2) + graph_ten_thousands
            | delete_zero + graph_lakhs
            | graph_ten_lakhs
        ).optimize()

        graph_crores = scale(digit, 7, suf_crore, suf_crore_gen, rem_crore)
        graph_ten_crores = scale(teens_and_ties, 7, suf_crore, suf_crore_gen, rem_crore)
        graph_hundred_crores = scale(graph_hundreds, 7, suf_crore, suf_crore_gen, rem_crore)
        self.graph_crores = graph_crores
        self.graph_ten_crores = graph_ten_crores
    
        def append_crore(coeff_remainder, with_sub_rem):
            g = coeff_remainder + (delete_zero**7) + suf_crore

            if with_sub_rem:
                g |= coeff_remainder + suf_crore_gen + insert_space + sub_crore_rem
            return g

        th_crore= scale(digit, 10, suf_thousand_crore, suf_thousand_crore_gen, rem_crore)
        graph_thousand_crores = (th_crore	 | append_crore(th_rem, with_sub_rem=True)).optimize()
     
        th_crore= scale(teens_and_ties, 10, suf_thousand_crore, suf_thousand_crore_gen, rem_crore)
        graph_ten_thousand_crores = (th_crore| append_crore(tth_rem, with_sub_rem=True)).optimize()

        th_crore= scale(digit, 12, suf_lakh_crore, suf_lakh_crore_gen, rem_crore)
        graph_lakh_crores = (th_crore| append_crore(l_rem, with_sub_rem=True)).optimize()

        th_crore= scale(teens_and_ties, 12, suf_lakh_crore, suf_lakh_crore_gen, rem_crore)
        graph_ten_lakh_crores = (th_crore	 | append_crore(tl_rem, with_sub_rem=True)).optimize()

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

        cardinal_with_leading_zeros = pynutil.add_weight(
            pynini.compose(NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph),
            0.5,
        )

        graph_no_commas = graph_without_leading_zeros | cardinal_with_leading_zeros
        delete_comma = pynutil.delete(",")

        western_format = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 3)
            + pynini.closure(delete_comma + pynini.closure(NEMO_ALL_DIGIT, 3, 3), 1))
        
        indian_format = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 2)
            + pynini.closure(
                delete_comma + pynini.closure(NEMO_ALL_DIGIT, 2, 2))+ delete_comma+ pynini.closure(NEMO_ALL_DIGIT, 3, 3))

        comma_number = western_format | indian_format

        cardinal_with_commas = pynutil.add_weight(pynini.compose(comma_number, graph_without_leading_zeros), 0.1)

        final_graph = graph_no_commas | cardinal_with_commas
        self.final_graph = final_graph.optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(final_graph)
