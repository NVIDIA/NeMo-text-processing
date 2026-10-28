# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.kn.graph_utils import (
    DIVIDE_WORD,
    MINUS_WORD,
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions
        fraction { numerator: "ಮೂರು" denominator: "ಸೊನ್ನೆ" } -> ಮೂರು ಬೈ ಸೊನ್ನೆ
        fraction { negative: "true" morphosyntactic_features: "ಅರ್ಧ" } -> ಮೈನಸ್ ಅರ್ಧ

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        words = pynini.string_file(get_abs_path("data/fraction/words.tsv"))
        zero = pynini.project(pynini.string_file(get_abs_path("data/numbers/zero.tsv")), "output")
        connector = pynini.project(pynini.accep("CONNECTOR") @ words, "output")

        def field(name, graph=pynini.closure(NEMO_NOT_QUOTE)):
            return pynutil.delete(name + ': "') + graph + pynutil.delete('"')

        sign = pynini.cross('negative: "true"', MINUS_WORD) + pynutil.insert(NEMO_SPACE) + pynutil.delete(NEMO_SPACE)
        integer = (
            field("integer_part") + pynutil.delete(NEMO_SPACE) + pynutil.insert(NEMO_SPACE + connector + NEMO_SPACE)
        )

        fraction = field("denominator") + NEMO_SPACE + field("numerator")

        divide_by_zero = (
            field("numerator")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.insert(NEMO_SPACE + DIVIDE_WORD + NEMO_SPACE)
            + field("denominator", zero)
        )

        special = field("morphosyntactic_features")

        normal = pynini.closure(integer, 0, 1) + (fraction | divide_by_zero)
        graph = pynini.closure(sign, 0, 1) + (normal | special)

        self.fst = self.delete_tokens(graph).optimize()
