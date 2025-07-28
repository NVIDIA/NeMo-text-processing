# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.fr.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinals
        e.g. ordinal { integer: "deux" morphosyntactic_features: "ième" } -> "deuxième"
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, strip_dashes: bool = False):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)

        ones = pynini.cross("un", "prem")

        irregular_numbers = pynini.string_file(get_abs_path("data/ordinals/irregular_numbers.tsv"))
        irregular_numbers = pynini.closure(pynini.closure(NEMO_NOT_QUOTE, 1) + NEMO_SPACE) + irregular_numbers
        exceptions = pynini.project(irregular_numbers, "input")
        exception_suffix = (
            pynutil.delete(" morphosyntactic_features: \"ième")
            + pynini.closure(pynini.accep("s"), 0, 1)
            + pynutil.delete("\"")
        )
        irregular_numbers_graph = (
            pynutil.delete("integer: \"") + irregular_numbers + pynutil.delete("\"") + exception_suffix
        )

        numbers = pynini.closure(NEMO_NOT_QUOTE, 1)
        numbers = pynini.difference(numbers, exceptions)

        if strip_dashes:
            remove_dashes = pynini.closure(NEMO_ALPHA, 1) + pynini.cross("-", " ") + pynini.closure(NEMO_ALPHA, 1)
            remove_dashes = pynini.closure(remove_dashes, 0)
            numbers = pynini.union(numbers, pynutil.add_weight(numbers @ remove_dashes, -0.0001))

        regular_ordinals = pynini.union(numbers, pynutil.add_weight(ones, -0.0001))
        regular_ordinals_graph = (
            pynutil.delete("integer: \"")
            + regular_ordinals
            + pynutil.delete("\"")
            + pynutil.delete(" morphosyntactic_features: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        final_graph = pynini.union(regular_ordinals_graph, irregular_numbers_graph)
        self.graph = final_graph

        self.fst = self.delete_tokens(final_graph).optimize()
