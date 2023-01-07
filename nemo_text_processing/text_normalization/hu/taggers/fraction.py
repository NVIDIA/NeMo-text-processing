# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.hu.utils import get_abs_path
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/6" ->
    fraction { integer: "huszonhárom" numerator: "négy" denominator: "hatod" preserve_order: true }

    Args:
        cardinal: cardinal GraphFst
        ordinal: ordinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal, ordinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        # ordinals are formed by adding -ik to the fractional, but we're working backwards
        # Except, for some reason, this does not work.
        # ordinal_graph = ordinal.bare_ordinals
        ord_endings = pynini.string_file(get_abs_path("data/ordinals/endings.tsv"))
        ord_exceptions = pynini.string_file(get_abs_path("data/ordinals/exceptional.tsv"))

        exceptions = pynini.string_map([("első", "egyed"), ("második", "fél")])
        # graph_fractional = (
        #     ordinal_graph
        #     @ pynini.cdrewrite(exceptions, "[BOS]", "[EOS]", NEMO_SIGMA)
        #     @ pynini.cdrewrite(pynutil.delete("ik"), "", "[EOS]", NEMO_SIGMA)
        # )
        graph_fractional = (
            (
                cardinal_graph
                @ pynini.cdrewrite(ord_exceptions, "[BOS]", "[EOS]", NEMO_SIGMA)
                @ pynini.cdrewrite(ord_endings, "", "[EOS]", NEMO_SIGMA)
            )
            @ pynini.cdrewrite(exceptions, "[BOS]", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(pynutil.delete("ik"), "", "[EOS]", NEMO_SIGMA)
        )
        if not deterministic:
            graph_fractional |= pynini.cross("2", "ketted")

        self.fractional = graph_fractional.optimize()

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
        )
        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.numerator = (
            pynutil.insert("numerator: \"") + cardinal_graph + pynini.cross(pynini.union("/", " / "), "\" ")
        )
        self.denominator = pynutil.insert("denominator: \"") + self.fractional + pynutil.insert("\"")

        self.graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(" "), 0, 1)
            + self.numerator
            + self.denominator
        )

        graph = self.graph + pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
