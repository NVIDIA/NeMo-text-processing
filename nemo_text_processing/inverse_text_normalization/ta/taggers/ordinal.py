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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_TA_LETTER, sequential


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying spoken ordinals, e.g.
        ஐந்தாவது -> ordinal { integer: "5" morphosyntactic_features: "வது" preserve_order: true }
        பத்தாம் -> ordinal { integer: "10" morphosyntactic_features: "ஆம்" preserve_order: true }
        ஐந்தாவதுக்கு -> ordinal { integer: "5" morphosyntactic_features: "வதுக்கு" preserve_order: true }

    Tamil reads the ordinal off its adjectival stem (ஐந்தா- -> ஐந்து -> 5) and carries the written
    marker in its own field rather than inverting the TN ordinal graph: the inflected tail is
    unbounded, so inverting it against the written marker leaves an ambiguous alignment the
    composition cannot merge.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="ordinal", kind="classify")

        # Undo the adjectival stem: the cardinal's final -உ becomes -ஆ and a ம்-final scale word
        # becomes -மா (ஐந்து -> ஐந்தா, ஆயிரம் -> ஆயிரமா, நூறு -> நூற்றா).
        to_cardinal = pynini.closure(NEMO_CHAR) + pynini.union(
            pynini.cross("ா", "ு"), pynini.cross("மா", "ம்"), pynini.cross("ற்றா", "று")
        )
        stem = sequential((to_cardinal @ cardinal.words_to_digits) | pynini.cross("முதலா", "1"))

        integer = pynutil.insert("integer: \"") + stem + pynutil.insert("\"")

        # -வது plus any inflected tail: ஐந்தாவது -> 5வது, ஐந்தாவதுக்கு -> 5வதுக்கு.
        graph_vathu = (
            integer
            + pynutil.insert(" morphosyntactic_features: \"")
            + pynini.accep("வத")
            + pynini.closure(NEMO_TA_LETTER, 1)
            + pynutil.insert("\"")
        )
        # The clipped ம் ordinal is written with the canonical ஆம் spelling (28ம் -> 28ஆம்).
        graph_aam = integer + pynutil.insert(" morphosyntactic_features: \"ஆம்\"") + pynutil.delete("ம்")

        graph = (graph_vathu | graph_aam) + pynutil.insert(" preserve_order: true")
        self.fst = self.add_tokens(graph).optimize()
