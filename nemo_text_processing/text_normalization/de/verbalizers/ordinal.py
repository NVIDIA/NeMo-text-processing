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
from nemo_text_processing.text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SIGMA, GraphFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing roman numerals
        e.g. ordinal { integer: "vier" } } -> "vierter"
                                           -> "viertes" ...

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv")).invert()
        graph_ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv")).invert()
        graph_thousands = pynini.string_file(get_abs_path("data/ordinals/thousands.tsv")).invert()

        graph = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        suffixes = pynini.union("ten", "tem", "ter", "tes", "te")
        convert_rest = pynutil.insert(suffixes, weight=0.01)
        self.ordinal_stem = graph_digit | graph_ties | graph_thousands

        suffix = pynini.cdrewrite(
            pynini.closure(self.ordinal_stem, 0, 1) + convert_rest, "", "[EOS]", NEMO_SIGMA,
        ).optimize()
        self.graph = pynini.compose(graph, suffix)
        self.suffix = suffix
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
