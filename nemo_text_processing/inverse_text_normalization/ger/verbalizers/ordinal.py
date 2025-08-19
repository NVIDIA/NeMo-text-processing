# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    delete_space,
    NEMO_DIGIT,
    NEMO_SPACE,
)


class OrdinalFst(GraphFst):
    """
    WFST for verbalizing ordinal numerals:

    ordinal { integer: "1" morphosyntactic_features: "./jahrhundert" } -> I Jh.
    ordinal { integer: "34" morphosyntactic_features: "." } -> 34.

    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")

        period = pynini.accep(".")

        # WFST mappings for Roman numerals
        digits = pynini.string_file(get_abs_path("data/ordinal/roman_digits.tsv"))
        tens = pynini.string_file(get_abs_path("data/ordinal/roman_tens.tsv"))
        zero = pynutil.delete("0")  # Roman numerals don't have a symbol for zero

        graph_integer = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT | period, 1)
            + pynutil.delete('"')
        )

        graph_morphosyntax = pynutil.delete(
            ' morphosyntactic_features: "'
        ) + pynini.accep(".")

        # WFST for Arabic numerals
        graph_arabic_numerals = graph_integer + graph_morphosyntax + pynutil.delete('"')

        # Filters for Roman digits
        map_one_digit = NEMO_DIGIT
        map_two_digits = NEMO_DIGIT**2

        # Converts from Arabic to Roman digits
        graph_single_digit_roman = map_one_digit @ digits
        graph_double_digit_roman = tens + (digits | zero)
        graph_double_digit_roman = map_two_digits @ graph_double_digit_roman
        graph_all_roman = graph_single_digit_roman | graph_double_digit_roman

        # WFST for Roman numerals
        graph_roman_numerals = (
            (graph_integer @ graph_all_roman)
            + graph_morphosyntax
            + pynini.cross("/", " ")
            # Uses "Jahrhundert/-e" to trigger conversion to Roman numerals
            + pynini.string_map([("Jahrhundert", "Jh."), ("Jahrhunderte", "Jh.")])
            # Applies optional era markers (B.C and A.D) following the century abbreviation
            + (
                pynini.accep(NEMO_SPACE)
                + (pynini.accep("v. Ch.") | pynini.accep("n. Ch."))
            ).ques
            + pynutil.delete('"').ques
        )

        # Build and optimze the final graph
        final_graph = graph_roman_numerals | graph_arabic_numerals
        delete_tokens = self.delete_tokens(final_graph)
        self.graph_ordinals = delete_tokens
        self.fst = delete_tokens.optimize()
