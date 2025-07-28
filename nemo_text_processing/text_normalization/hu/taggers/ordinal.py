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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.hu.taggers.cardinal import filter_punctuation
from nemo_text_processing.text_normalization.hu.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "2." -> ordinal { integer: "második" } }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        endings = pynini.string_file(get_abs_path("data/ordinals/endings.tsv"))
        exceptions = pynini.string_file(get_abs_path("data/ordinals/exceptional.tsv"))
        superessive_endings = pynini.string_file(get_abs_path("data/ordinals/superessive_endings.tsv"))
        superscript2digit = pynini.string_file(get_abs_path("data/ordinals/superscript_digits.tsv"))

        cardinal_graph = cardinal.graph
        bare_ordinals = (
            cardinal_graph
            @ pynini.cdrewrite(exceptions, "[BOS]", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(endings, "", "[EOS]", NEMO_SIGMA)
        )
        self.bare_ordinals = bare_ordinals
        self.filtered_ordinals = filter_punctuation(bare_ordinals).optimize()

        self.superessive = self.bare_ordinals @ pynini.cdrewrite(superessive_endings, "", "[EOS]", NEMO_SIGMA)
        self.superscript_to_superessive = pynini.closure(superscript2digit) @ self.superessive

        self.graph = pynini.union(
            self.filtered_ordinals + pynutil.delete("."), self.filtered_ordinals.project("output")
        ).optimize()

        # For some reason, bare_ordinals does not work when exported, so doing this here
        fractional_exceptions = pynini.string_map([("első", "egyed"), ("második", "fél")])
        self.fractional = (
            bare_ordinals
            @ pynini.cdrewrite(fractional_exceptions, "[BOS]", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(pynutil.delete("ik"), "", "[EOS]", NEMO_SIGMA)
        ).optimize()
        if not deterministic:
            self.fractional |= pynini.cross("2", "ketted")

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
