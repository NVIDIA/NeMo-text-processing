# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.se.graph_utils import SE_UPPER
from nemo_text_processing.text_normalization.se.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.se.utils import get_abs_path


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word.
        e.g. hund -> tokens { name: "hund" }

    Args:
        cardinal: cardinal graph used to verbalize digit runs in mixed strings
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)
        word = pynutil.insert("name: \"") + pynini.closure(NEMO_NOT_SPACE, 1) + pynutil.insert("\"")
        self.fst = word.optimize()

        digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
        zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        single_digit = digit | zero
        space = pynutil.insert(" ")

        leading_zero = ("0" + pynini.closure(NEMO_DIGIT, 1)) @ cardinal.graph_with_leading_zero
        up_to_three = ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0, 2)) @ cardinal.graph_with_leading_zero
        four_digits = (
            ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ cardinal.graph_with_leading_zero
            + space
            + ((NEMO_DIGIT**2) @ cardinal.graph_with_leading_zero)
        )
        long_digits = single_digit + pynini.closure(space + single_digit, 4)
        digits = leading_zero | up_to_three | four_digits | long_digits
        letters = SE_UPPER + pynini.closure(space + SE_UPPER)

        alphanumeric = pynini.union(
            letters
            + space
            + digits
            + pynini.closure(space + letters + space + digits)
            + pynini.closure(space + letters, 0, 1),
            digits
            + space
            + letters
            + pynini.closure(space + digits + space + letters)
            + pynini.closure(space + digits, 0, 1),
        )
        alphanumeric = pynini.cdrewrite(pynutil.delete("-"), "", "", NEMO_SIGMA) @ alphanumeric
        self.alphanumeric = pynutil.insert('name: "') + convert_space(alphanumeric) + pynutil.insert('"')
