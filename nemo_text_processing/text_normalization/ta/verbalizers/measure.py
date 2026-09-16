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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ta.graph_utils import MINUS_WORD


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measures, e.g.
        measure { cardinal { integer: "ஐந்து" } units: "கிலோமீட்டர்" preserve_order: true } -> ஐந்து கிலோமீட்டர்
        measure { decimal { integer_part: "பன்னிரண்டு" fractional_part: "ஐந்து" } units: "கிலோகிராம்" preserve_order: true } -> பன்னிரண்டு புள்ளி ஐந்து கிலோகிராம்
        measure { cardinal { integer: "ஒன்று" } units: "கிலோகிராம்" preserve_order: true } -> ஒரு கிலோகிராம்

    Args:
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", f"{MINUS_WORD} ") + delete_space, 0, 1)
        # A whole-field ஒன்று before the unit noun reads as ஒரு.
        one_as_oru = pynini.cross("ஒன்று", "ஒரு") | pynini.difference(
            pynini.closure(NEMO_NOT_QUOTE, 1), pynini.accep("ஒன்று")
        )
        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + pynutil.delete("integer: \"")
            + one_as_oru
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
        )
        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + decimal.integer
            + delete_space
            + decimal.fractional
            + delete_space
            + pynutil.delete("}")
        )
        units = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        graph = (
            optional_sign
            + (graph_cardinal | graph_decimal)
            + delete_space
            + insert_space
            + units
            + delete_preserve_order
        )
        self.fst = self.delete_tokens(graph).optimize()
