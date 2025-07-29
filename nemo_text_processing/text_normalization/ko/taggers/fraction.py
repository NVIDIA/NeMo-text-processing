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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Korean fractions, e.g.
    3/5 → tokens { fraction { numerator: "삼" denominator: "오" } }
    2과7/9 → tokens { fraction { integer_part: "이" numerator: "칠" denominator: "구" } }
    마이너스3/5 → tokens { fraction { negative: "마이너스" numerator: "삼" denominator: "오" } }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal = cardinal.graph
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        slash = pynutil.delete('/')
        morphemes = pynini.accep('분의')
        root = pynini.accep('√')

        # Decimal number (e.g., 1.23 → 일점이삼)
        decimal_number = cardinal + pynini.cross(".", "점") + pynini.closure(graph_digit | graph_zero)

        # Accept cardinal / root + cardinal / decimal / root + decimal
        numeral = cardinal | (root + cardinal) | decimal_number | (root + decimal_number)

        # Integer part (e.g., 2과, 1와)
        integer_component = (
            pynutil.insert('integer_part: "')
            + numeral
            + (pynini.accep("과") | pynini.accep("와"))
            + pynutil.insert('"')
        )

        integer_component_with_space = integer_component + pynutil.insert(NEMO_SPACE)

        # Denominator and numerator
        denominator_component = pynutil.insert('denominator: "') + numeral + pynutil.insert('"')

        numerator_component = pynutil.insert('numerator: "') + numeral + pynutil.insert('"')

        # Format 1: 3/4 style
        graph_fraction_slash = (
            pynini.closure(integer_component_with_space, 0, 1)
            + numerator_component
            + slash
            + pynutil.insert(NEMO_SPACE)
            + denominator_component
        )

        # Format 2: Korean native "4분의3" style
        graph_fraction_word = (
            pynini.closure(integer_component_with_space, 0, 1)
            + denominator_component
            + pynutil.delete("분의")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('morphosyntactic_features: "분의"')
            + pynutil.insert(NEMO_SPACE)
            + numerator_component
        )

        # Optional minus sign
        optional_sign = (
            pynutil.insert('negative: "')
            + (pynini.accep("마이너스") | pynini.cross("-", "마이너스"))
            + pynutil.insert('"')
            + pynutil.insert(NEMO_SPACE)
        )

        # Combine full graph
        graph = pynini.closure(optional_sign, 0, 1) + (graph_fraction_slash | graph_fraction_word)
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
