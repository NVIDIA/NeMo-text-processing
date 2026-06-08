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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese fraction numbers, e.g.
        fraction { negative: "true" integer_part: "hai mươi ba" numerator: "một" denominator: "năm" } -> âm hai mươi ba và một phần năm
        fraction { numerator: "ba" denominator: "chín" } -> ba phần chín
        fraction { integer_part: "một trăm" numerator: "hai" denominator: "ba" } -> một trăm và hai phần ba

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.cross("negative: \"true\"", "âm ")
        if not deterministic:
            optional_sign |= pynini.cross("negative: \"true\"", "trừ ")
        optional_sign = pynini.closure(optional_sign + delete_space, 0, 1)

        part = pynini.closure(NEMO_NOT_QUOTE)
        delete_quotes = delete_space + pynutil.delete("\"") + part + pynutil.delete("\"")

        integer_tagged = pynutil.delete("integer_part:") + delete_quotes
        numerator_tagged = pynutil.delete("numerator:") + delete_quotes
        denominator_tagged = pynutil.delete("denominator:") + delete_quotes

        fraction_part = numerator_tagged + delete_space + pynutil.insert(" phần ") + denominator_tagged

        simple_fraction = fraction_part
        mixed_fraction = integer_tagged + delete_space + pynutil.insert(" và ") + fraction_part

        self.numbers = optional_sign + (simple_fraction | mixed_fraction)

        self.fst = self.delete_tokens(self.numbers).optimize()
