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
    delete_preserve_order,
    delete_space,
    insert_space,
)


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing Roman numerals read in context, e.g.
        roman { key_cardinal: "வகுப்பு" integer: "பன்னிரண்டு" preserve_order: true } -> வகுப்பு பன்னிரண்டு
        roman { integer: "பன்னிரண்டாம்" key_cardinal: "வகுப்பு" preserve_order: true } -> பன்னிரண்டாம் வகுப்பு
        roman { integer: "பன்னிரண்டாம்" } -> பன்னிரண்டாம்

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)

        # A multi-word cue travels with U+00A0 NO-BREAK SPACE; speak it with plain spaces.
        key = (
            pynutil.delete("key_cardinal: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        ) @ pynini.cdrewrite(pynini.cross(" ", " "), "", "", NEMO_SIGMA)
        integer = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph = pynini.union(
            key + delete_space + insert_space + integer,
            integer + delete_space + insert_space + key,
            integer,
        )
        self.fst = self.delete_tokens(graph + delete_preserve_order).optimize()
