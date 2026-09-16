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
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import ambiguous_words, kept_scale_words
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path, load_rows
from nemo_text_processing.text_normalization.en.graph_utils import convert_space
from nemo_text_processing.text_normalization.ta.graph_utils import CURRENCY_SYMBOLS, NEMO_ALL_DIGIT, NEMO_TA_LETTER


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying spans that must pass through ITN unchanged, e.g.
        எல்லாம் ஒன்று -> tokens { name: "எல்லாம் ஒன்று" }
        கால் வலிக்கிறது -> tokens { name: "கால் வலிக்கிறது" }
        ₹5 கோடி -> tokens { name: "₹5 கோடி" }

    Three kinds of span: the phrases of ``data/whitelist/prose_phrases.tsv``, where a numeral
    is a pronoun or an idiom; a ``standalone`` word of ``data/numbers/ambiguous.tsv`` (கால்
    "leg", அரை "room") before another Tamil word, which is prose rather than a fraction; and an
    already-written number (12.5%, 10-20, +91 9876543210, ₹5 கோடி, 2024ல்), which must not be
    split into punctuation and digits or re-read.

    Args:
        input_file: path to a file with whitelist replacements (each line: spoken\twritten),
            added to the default spans
    """

    def __init__(self, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")

        phrases = [row[0] for row in load_rows(get_abs_path("data/whitelist/prose_phrases.tsv"), 1)]
        prose = pynini.union(*phrases)

        standalone = [word for word, _ in ambiguous_words("standalone")]
        followed = pynini.union(*standalone) + pynini.accep(" ") + pynini.closure(NEMO_TA_LETTER, 1)

        # A sign, a currency symbol, a glued case suffix (hyphenated or not) and a kept scale word
        # all travel with the digits of a written number.
        written = (
            pynini.closure(pynini.union("-", "+"), 0, 1)
            + pynini.closure(pynini.union(*CURRENCY_SYMBOLS), 0, 1)
            + pynini.closure(NEMO_ALL_DIGIT, 1)
            + pynini.closure(pynini.union(*".:,/-") + pynini.closure(NEMO_ALL_DIGIT, 1))
            + pynini.closure("%", 0, 1)
            + pynini.closure(pynini.closure("-", 0, 1) + pynini.closure(NEMO_TA_LETTER, 1), 0, 1)
            + pynini.closure(" " + pynini.union(*kept_scale_words()), 0, 1)
        )

        # A written number outranks every reading; the prose spans only need to beat the number
        # classes, which the tokenizer weights above 1.0.
        graph = pynutil.add_weight(written, -0.2) | prose | followed
        if input_file:
            graph |= pynini.string_map([row[:2] for row in load_rows(input_file, 2)])

        self.graph = convert_space(graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
