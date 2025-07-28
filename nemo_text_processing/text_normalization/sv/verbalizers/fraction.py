# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SIGMA, GraphFst, insert_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. tokens { fraction { integer: "tjugotre" numerator: "fyra" denominator: "femtedel" } } ->
        tjugotre och fyra femtedelar

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)
        plurals = pynini.string_map([("kvart", "kvartar"), ("halv", "halva"), ("del", "delar")])

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")

        denominators_sg = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        denominators_pl = (
            pynutil.delete("denominator: \"")
            + (pynini.closure(NEMO_NOT_QUOTE) @ pynini.cdrewrite(plurals, "", "[EOS]", NEMO_SIGMA))
            + pynutil.delete("\"")
        )
        self.denominators = denominators_sg | denominators_pl

        either_one = pynini.union("en", "ett")
        numerator_one = pynutil.delete("numerator: \"") + pynutil.delete(either_one) + pynutil.delete("\" ")
        if not deterministic:
            numerator_one |= pynutil.delete("numerator: \"") + either_one + pynutil.delete("\" ") + insert_space
        numerator_rest = (
            pynutil.delete("numerator: \"")
            + (
                (pynini.closure(NEMO_NOT_QUOTE) - either_one)
                @ pynini.cdrewrite(pynini.cross("ett", "en"), "[BOS]", "[EOS]", NEMO_SIGMA)
            )
            + pynutil.delete("\" ")
        )

        graph_sg = numerator_one + denominators_sg
        graph_pl = numerator_rest + insert_space + denominators_pl
        graph = graph_sg | graph_pl

        conjunction = pynutil.insert("och ")
        if not deterministic and not lm:
            conjunction = pynini.closure(conjunction, 0, 1)

        integer = pynini.closure(integer + insert_space + conjunction, 0, 1)

        graph = integer + graph

        self.graph = graph
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
