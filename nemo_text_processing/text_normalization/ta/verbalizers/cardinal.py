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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from nemo_text_processing.text_normalization.ta.graph_utils import MINUS_WORD, PLUS_WORD


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinals, e.g.
        cardinal { negative: "true" integer: "இருபத்துமூன்று" } -> மைனஸ் இருபத்துமூன்று
        cardinal { positive: "true" integer: "ஐந்து" } -> பிளஸ் ஐந்து

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        sign = pynini.cross("negative: \"true\"", f"{MINUS_WORD} ") | pynini.cross(
            "positive: \"true\"", f"{PLUS_WORD} "
        )
        self.optional_sign = pynini.closure(sign + delete_space, 0, 1)
        self.integer = delete_space + pynutil.delete("\"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        self.numbers = self.optional_sign + pynutil.delete("integer:") + self.integer
        self.fst = self.delete_tokens(self.numbers).optimize()
