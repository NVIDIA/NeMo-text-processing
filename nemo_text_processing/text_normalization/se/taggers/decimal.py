# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023, 2026, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.se.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.se.utils import get_abs_path


class DecimalFst(GraphFst):
    """Classifies Northern Sámi decimal numbers."""

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
        zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        single_digit = digit | zero
        spoken_space = pynutil.insert(" ")

        short_fraction = pynini.closure(NEMO_DIGIT, 1, 3) @ cardinal.graph_with_leading_zero
        long_fraction = single_digit + pynini.closure(spoken_space + single_digit, 3)
        comma_fraction = pynutil.delete(",") + (short_fraction | long_fraction)
        dot_fraction = pynutil.delete(".") + (
            (NEMO_DIGIT | NEMO_DIGIT**2) @ cardinal.graph_with_leading_zero | long_fraction
        )

        integer = pynutil.insert('integer_part: "') + cardinal.graph_with_leading_zero + pynutil.insert('" ')
        fractional = pynutil.insert('fractional_part: "') + (comma_fraction | dot_fraction) + pynutil.insert('"')
        negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        graph = negative + integer + fractional
        self.fst = self.add_tokens(graph).optimize()
