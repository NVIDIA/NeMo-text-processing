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
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_zero_or_one_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import RANGE_WORD
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

# Single letters that are far more often part of an identifier (47A, 5G) than a unit.
ID_PRONE = frozenset("ABCGJKNVWXbdhqsx*")


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measures, e.g.
        5 கி.மீ. -> measure { cardinal { integer: "ஐந்து" } units: "கிலோமீட்டர்" preserve_order: true }
        12.5kg -> measure { decimal { integer_part: "பன்னிரண்டு" fractional_part: "ஐந்து" } units: "கிலோகிராம்" preserve_order: true }
        -40°C -> measure { negative: "true" cardinal { integer: "நாற்பது" } units: "டிகிரி செல்சியஸ்" preserve_order: true }
        5-10 kg -> measure { cardinal { integer: "ஐந்து முதல் பத்து" } units: "கிலோகிராம்" preserve_order: true }

    Reads ``data/measure/unit.tsv``; a single-letter unit needs a space before it, so a glued
    identifier such as 47A is left to the serial class.

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        rows = [r for r in load_labels(get_abs_path("data/measure/unit.tsv")) if len(r) >= 2]
        multi = pynini.string_map([(k, v) for k, v, *_ in rows if len(k) > 1 or k not in ID_PRONE]).optimize()
        single = pynini.string_map([(k, v) for k, v, *_ in rows if len(k) == 1]).optimize()

        # Accept uppercase spellings of Latin units (5KG).
        lowercase = pynini.closure(TO_LOWER | pynini.union(*"abcdefghijklmnopqrstuvwxyz°²./"), 2)
        multi |= pynini.compose(lowercase, multi).optimize()

        unit_multi = convert_space(multi).optimize()
        unit_single = convert_space(single).optimize()
        unit_part = (delete_zero_or_one_space + unit_multi) | (pynutil.delete(" ") + unit_single)

        # 5-10 kg reads as a range amount.
        amount = pynini.union(
            cardinal.final_graph, cardinal.final_graph + pynini.cross("-", f" {RANGE_WORD} ") + cardinal.final_graph
        ).optimize()
        graph_cardinal = pynutil.insert("cardinal { integer: \"") + amount + pynutil.insert("\" }")
        graph_decimal = (
            pynutil.insert("decimal { ")
            + decimal.graph_integer
            + pynutil.delete(".")
            + insert_space
            + pynutil.insert("fractional_part: \"")
            + decimal.graph
            + pynutil.insert("\" }")
        )

        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph = (
            optional_negative
            + (graph_cardinal | graph_decimal)
            + pynutil.insert(" units: \"")
            + unit_part
            + pynutil.insert("\"")
            + pynutil.insert(" preserve_order: true")
        )
        self.fst = self.add_tokens(graph).optimize()
