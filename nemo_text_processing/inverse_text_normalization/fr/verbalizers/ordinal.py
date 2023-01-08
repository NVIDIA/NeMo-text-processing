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
from nemo_text_processing.inverse_text_normalization.fr.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
        ordinal { integer: "13" morphosyntactic_features: "e" } -> 13ᵉ

    Given 'special' terms for ordinals (e.g. siècle), renders
        amount in conventional format. e.g.

        ordinal { integer: "13" morphosyntactic_features: "e/siècle" } -> XIIIᵉ
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph_integer = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        replace_suffix = pynini.union(
            pynini.cross("e", "ᵉ"),  # only delete first quote since there may be more features
            pynini.cross("d", "ᵈ"),
            pynini.cross("r", "ʳ"),
            pynini.cross("s", "ˢ"),
        )
        replace_suffix = pynutil.delete(" morphosyntactic_features: \"") + replace_suffix.plus

        graph_arabic = graph_integer + replace_suffix.plus

        # For roman.
        graph_roman_digits = pynini.string_file(get_abs_path("data/roman/digits_large.tsv")).invert()
        graph_roman_ties = pynini.string_file(get_abs_path("data/roman/ties_large.tsv")).invert()
        graph_roman_hundreds = pynini.string_file(get_abs_path("data/roman/hundreds_large.tsv")).invert()
        graph_roman_zero_digit = pynutil.delete("0")

        graph_roman_hundreds = NEMO_DIGIT ** 3 @ (
            graph_roman_hundreds
            + pynini.union(graph_roman_ties, graph_roman_zero_digit)
            + pynini.union(graph_roman_digits, graph_roman_zero_digit)
        )
        graph_roman_ties = NEMO_DIGIT ** 2 @ (
            graph_roman_ties + pynini.union(graph_roman_digits, graph_roman_zero_digit)
        )
        graph_roman_digits = NEMO_DIGIT @ graph_roman_digits

        graph_roman_integers = graph_roman_hundreds | graph_roman_ties | graph_roman_digits

        graph_roman = (graph_integer @ graph_roman_integers) + replace_suffix
        graph_roman += pynini.cross("/", " ") + "siècle"

        graph = (graph_roman | graph_arabic) + pynutil.delete("\"")

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
