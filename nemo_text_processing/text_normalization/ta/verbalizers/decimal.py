# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    MINUS_WORD, NEMO_NOT_QUOTE, POINT, GraphFst, delete_space,
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimals
        e.g. decimal { integer_part: "ஒன்று" fractional_part: "புள்ளி ஐந்து" } -> "ஒன்று புள்ளி ஐந்து"
        e.g. decimal { negative: "true" integer_part: "இரண்டு" fractional_part: "புள்ளி ஆறு ஏழு" } -> "கழித்தல் இரண்டு புள்ளி ஆறு ஏழு"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """
    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        def quoted_field(name):
            return (pynutil.delete(f'{name}: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')).optimize()

        sign_with_space = pynini.closure(
            pynini.cross('negative: "true" ', MINUS_WORD) + pynutil.insert(" "), 0, 1
        ).optimize()

        with_integer = (
            sign_with_space + quoted_field("integer_part") + delete_space
            + pynutil.insert(f" {POINT} ") + quoted_field("fractional_part")
        ).optimize()

        without_integer = (
            sign_with_space + pynutil.delete('has_integer: "false" ')
            + pynutil.insert(f"{POINT} ") + quoted_field("fractional_part")
        ).optimize()

        literal_point_decimal = (
            sign_with_space + quoted_field("integer_part") + delete_space
            + pynutil.delete('literal_point: "true" ')
            + pynutil.insert(" . ")
            + quoted_field("fractional_part")
        ).optimize()

        self.fst = self.delete_tokens(with_integer | without_integer | literal_point_decimal).optimize()
