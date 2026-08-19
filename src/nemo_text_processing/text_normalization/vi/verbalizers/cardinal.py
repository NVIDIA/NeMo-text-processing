# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese cardinal numbers, e.g.
        cardinal { negative: "true" integer: "hai mươi ba" } -> âm hai mươi ba
        cardinal { integer: "một trăm" } -> một trăm

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        # Handle negative sign - Vietnamese uses "âm" for negative numbers
        self.optional_sign = pynini.cross("negative: \"true\"", "âm ")
        if not deterministic:
            # Alternative ways to say negative in Vietnamese
            self.optional_sign |= pynini.cross("negative: \"true\"", "trừ ")
            self.optional_sign |= pynini.cross("negative: \"true\"", "âm ")

        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)

        # Handle the integer part
        integer = pynini.closure(NEMO_NOT_QUOTE)

        self.integer = delete_space + pynutil.delete("\"") + integer + pynutil.delete("\"")
        integer = pynutil.delete("integer:") + self.integer

        # Combine negative sign with integer
        self.numbers = self.optional_sign + integer

        # Delete the token structure and create final FST
        delete_tokens = self.delete_tokens(self.numbers)
        self.fst = delete_tokens.optimize()
