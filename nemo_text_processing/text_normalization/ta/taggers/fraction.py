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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_ALL_DIGIT, NEMO_ALL_ZERO, VULGAR_PAIRS


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions, e.g.
        3/4 -> fraction { numerator: "மூன்று" denominator: "நான்கு" }
        2 3/4 -> fraction { integer_part: "இரண்டு" numerator: "மூன்று" denominator: "நான்கு" }
        ½ -> fraction { numerator: "ஒன்று" denominator: "இரண்டு" }

    A vulgar sign is tagged as its numerator and denominator words, so the verbalizer speaks
    it as the everyday word (அரை) exactly as it does 1/2.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        any_digit = NEMO_ALL_DIGIT

        # A zero or zero-led denominator (1/0, 15/06) is not a fraction.
        non_zero_led = pynini.difference(
            pynini.closure(any_digit, 1), NEMO_ALL_ZERO + pynini.closure(any_digit)
        ).optimize()
        denominator_graph = pynini.compose(non_zero_led, cardinal_graph).optimize()

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        # A zero-led numerator (06/24) is a date fragment, not a fraction.
        numerator_input = pynini.difference(
            pynini.closure(any_digit, 1), NEMO_ALL_ZERO + pynini.closure(any_digit, 1)
        ).optimize()
        numerator = (
            pynutil.insert("numerator: \"")
            + pynini.compose(numerator_input, cardinal_graph)
            + (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" "))
        )
        denominator = pynutil.insert("denominator: \"") + denominator_graph + pynutil.insert("\"")

        graph = pynini.closure(integer + pynini.accep(" "), 0, 1) + numerator + denominator
        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        graph = optional_negative + graph

        # Vulgar signs, alone or after an integer (2¾, 12 ½).
        optional_space = pynutil.delete(pynini.closure(" ", 0, 1))
        pairs = pynini.union(
            *[
                pynutil.delete(sign) + pynutil.insert(f"numerator: \"{num}\" denominator: \"{den}\"")
                for sign, (num, den) in VULGAR_PAIRS.items()
            ]
        )
        graph |= optional_negative + (pynini.closure(integer + optional_space + pynutil.insert(" "), 0, 1) + pairs)

        self.graph = graph
        self.fst = self.add_tokens(self.graph).optimize()
