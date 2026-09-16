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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, GraphFst, convert_space, insert_space
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_TA_LETTER
from nemo_text_processing.text_normalization.ta.taggers.ordinal import first_ordinal
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

MAX_NUMERAL = 39

_ROMAN_VALUES = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def to_roman(n: int) -> str:
    """
    The Roman numeral for ``n`` (1-39).
    """
    letters = []
    for value, symbol in _ROMAN_VALUES:
        while n >= value:
            letters.append(symbol)
            n -= value
    return "".join(letters)


def roman_to_digits() -> 'pynini.FstLike':
    """
    Transducer from a Roman numeral I-XXXIX to its ASCII digits.
    """
    return pynini.string_map([(to_roman(n), str(n)) for n in range(1, MAX_NUMERAL + 1)]).optimize()


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying Roman numerals in context, e.g.
        வகுப்பு XII -> roman { key_cardinal: "வகுப்பு" integer: "பன்னிரண்டு" preserve_order: true }
        XII வகுப்பு -> roman { integer: "பன்னிரண்டாம்" key_cardinal: "வகுப்பு" preserve_order: true }
        ராஜராஜன்-II -> roman { key_cardinal: "ராஜராஜன்" integer: "இரண்டு" preserve_order: true }
        XIIஆம் -> roman { integer: "பன்னிரண்டாம்" }

    A Roman numeral is read as a number only where context says so: a cue word before or after
    it, a name joined to it with a hyphen, or a written ordinal marker glued to it. A bare I,
    V, X or MIX is a word. The cue words come from ``data/roman/context.tsv`` (cue word ->
    the written ordinal marker the numeral takes when it precedes the cue).

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        to_digits = roman_to_digits()
        cardinal_words = pynini.compose(to_digits, cardinal.final_graph).optimize()
        ordinal_reader = pynini.union(
            cardinal.ordinal_graph(cardinal.final_graph), pynutil.add_weight(first_ordinal(), -0.1)
        ).optimize()

        def field(name: str, value: 'pynini.FstLike') -> 'pynini.FstLike':
            return pynutil.insert(f"{name}: \"") + value + pynutil.insert("\"")

        separator = pynutil.delete(pynini.union(" ", "-")) + insert_space
        cued = []
        for cue, marker, *_ in load_labels(get_abs_path("data/roman/context.tsv")):
            key = field("key_cardinal", convert_space(pynini.accep(cue)))
            cued.append(key + separator + field("integer", cardinal_words))
            ordinal = pynini.compose(to_digits + pynutil.insert(marker), ordinal_reader)
            cued.append(field("integer", ordinal) + separator + key)
        name = field("key_cardinal", pynini.closure(pynini.union(NEMO_TA_LETTER, NEMO_ALPHA), 1))
        cued.append(name + pynutil.delete("-") + insert_space + field("integer", cardinal_words))
        in_context = pynini.union(*cued) + pynutil.insert(" preserve_order: true")

        glued = pynini.compose(to_digits + pynini.closure(NEMO_TA_LETTER, 1), ordinal_reader)
        graph = pynini.union(in_context, field("integer", glued))
        self.fst = self.add_tokens(graph).optimize()
