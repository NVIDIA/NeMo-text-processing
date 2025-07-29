# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Korean fractions, e.g.
    tokens { fraction { numerator: "3" denominator: "5" } } → 5분의3
    tokens { fraction { integer_part: "2" numerator: "7" denominator: "9" } } → 2과 9분의7
    tokens { fraction { denominator: "√8" numerator: "4" } } → 루트8분의4
    tokens { fraction { denominator: "2.75" numerator: "125" } } → 2.75분의125
    tokens { fraction { negative: "마이너스" numerator: "10" denominator: "11" } } → 마이너스11분의10
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Handles square root symbols like "√3" → "루트3"
        denominator_root = pynini.cross("√", "루트") + pynini.closure(NEMO_NOT_QUOTE)
        numerator_root = pynini.cross("√", "루트") + pynini.closure(NEMO_NOT_QUOTE)

        # Matches non-root numeric content
        denominator = pynini.closure(NEMO_NOT_QUOTE - "√")
        numerator = pynini.closure(NEMO_NOT_QUOTE - "√")

        # Delete FST field: denominator and extract value
        denominator_component = pynutil.delete('denominator: "') + (denominator_root | denominator) + pynutil.delete('"')
        numerator_component = pynutil.delete('numerator: "') + (numerator_root | numerator) + pynutil.delete('"')

        # Match fraction form: "denominator + 분의 + numerator"
        # Also deletes optional morphosyntactic_features: "분의" if present
        graph_fraction = (
            denominator_component
            + pynutil.delete(NEMO_SPACE)
            + pynini.closure(pynutil.delete('morphosyntactic_features:') + delete_space + pynutil.delete('"분의"') + delete_space, 0, 1)
            + pynutil.insert("분의")
            + numerator_component
        )

        # Match and delete integer_part field (e.g., "2" in "2과3분의1")
        graph_integer = (
            pynutil.delete('integer_part:') + delete_space + pynutil.delete('"')
            + pynini.closure(pynini.union("√", ".", NEMO_NOT_QUOTE - '"'))
            + pynutil.delete('"')
        )
        graph_integer_fraction = graph_integer + delete_space + graph_fraction

        # Match and delete optional negative field (e.g., "마이너스")
        optional_sign = (
            pynutil.delete('negative:') + delete_space + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE - '"') + pynutil.delete('"') + delete_space
        )

        # Final graph handles optional negative + (integer + fraction | fraction only)
        graph = pynini.closure(optional_sign, 0, 1) + (
            graph_integer_fraction | graph_fraction
        )

        # Final optimized verbalizer FST
        final_graph = self.delete_tokens(graph)
        self.fst = final_graph.optimize()
