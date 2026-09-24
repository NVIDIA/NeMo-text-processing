# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space


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
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        delete_space = pynutil.delete(" ")
        self.optional_sign = pynini.closure(
            pynini.cross("negative: \"true\"", "கழித்தல்") + delete_space + insert_space, 0, 1
        )
        self.integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        self.fractional = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        graph = self.optional_sign + self.integer + delete_space + insert_space + self.fractional

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
