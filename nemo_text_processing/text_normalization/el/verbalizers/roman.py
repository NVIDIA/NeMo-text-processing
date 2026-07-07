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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing roman numerals in Greek,
        e.g. tokens { roman { integer: "ένα" } } -> ένα

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)

        cardinal = pynini.closure(NEMO_NOT_QUOTE)

        # key_cardinal: "Κεφάλαιο" integer: "πέντε" -> Κεφάλαιο πέντε
        graph = (
            pynutil.delete("key_cardinal: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" ")
            + pynutil.delete("integer: \"")
            + cardinal
            + pynutil.delete("\"")
        ).optimize()

        # default_cardinal: "default" integer: "δέκα" -> δέκα
        graph |= (
            pynutil.delete("default_cardinal: \"default\" integer: \"") + cardinal + pynutil.delete("\"")
        ).optimize()

        # default_ordinal: "default" integer: "πέντε" -> πέντε
        graph |= (
            pynutil.delete("default_ordinal: \"default\" integer: \"") + cardinal + pynutil.delete("\"")
        ).optimize()

        # key_the_ordinal: "Ελισάβετ" integer: "δύο" -> Ελισάβετ δύο
        graph |= (
            pynutil.delete("key_the_ordinal: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" ")
            + pynutil.delete("integer: \"")
            + cardinal
            + pynutil.delete("\"")
        ).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
