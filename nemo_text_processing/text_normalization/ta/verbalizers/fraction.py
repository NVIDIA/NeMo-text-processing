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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import FRACTION_WORD, MINUS_WORD, TA_ARAI
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions, e.g.
        fraction { numerator: "ஒன்று" denominator: "இரண்டு" } -> அரை
        fraction { numerator: "ஐந்து" denominator: "எழுபத்தேழு" } -> ஐந்து கீழ் எழுபத்தேழு
        fraction { integer_part: "இரண்டு" numerator: "மூன்று" denominator: "நான்கு" } -> இரண்டே முக்கால்

    1/2, 1/4 and 3/4 have their own everyday words (``data/fraction/idiomatic.tsv``) and are
    spoken as those; any other pair is read with கீழ். A mixed number fuses with an everyday
    fraction (2 3/4 -> இரண்டே முக்கால், 1 1/2 -> ஒன்றரை) and otherwise joins with மற்றும்.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        denominator = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        special_rows = [r for r in load_labels(get_abs_path("data/fraction/idiomatic.tsv")) if len(r) >= 3]

        # Both readings consume the same field order so the weight decides between them, because
        # the engine picks a field permutation before the verbalizer sees the token.
        special = pynini.union(
            *[
                pynutil.delete("numerator: \"")
                + pynutil.delete(num)
                + pynutil.delete("\"")
                + delete_space
                + pynutil.delete("denominator: \"")
                + pynini.cross(den, word)
                + pynutil.delete("\"")
                for num, den, word, *_ in special_rows
            ]
        ).optimize()
        with_keezh = numerator + delete_space + insert_space + pynutil.insert(FRACTION_WORD + " ") + denominator

        # The half joins as -ரை; the quarters take the -ஏ link and stay a separate word. An
        # integer that does not end in -உ (ஆயிரம்) has no fused form and falls back to மற்றும்.
        mixed = []
        for num, den, word, *_ in special_rows:
            link, tail = ("ரை", "") if word == TA_ARAI else ("ே", " " + word)
            mixed.append(
                pynutil.delete("integer_part: \"")
                + (pynini.closure(NEMO_NOT_QUOTE, 1) @ (NEMO_SIGMA + pynini.cross("ு", link)))
                + pynutil.delete("\"")
                + delete_space
                + pynutil.delete("numerator: \"")
                + pynutil.delete(num)
                + pynutil.delete("\"")
                + delete_space
                + pynutil.delete("denominator: \"")
                + pynini.cross(den, tail)
                + pynutil.delete("\"")
            )

        bare = pynutil.add_weight(special, -1.0) | with_keezh
        graph = (
            bare
            | pynutil.add_weight(pynini.union(*mixed), -2.0)
            | integer + delete_space + pynutil.insert(" மற்றும் ") + bare
        )

        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", f"{MINUS_WORD} ") + delete_space, 0, 1)
        self.graph = optional_sign + graph
        self.fst = self.delete_tokens(self.graph).optimize()
