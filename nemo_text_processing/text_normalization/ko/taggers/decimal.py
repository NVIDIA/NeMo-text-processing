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


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers in Korean, e.g.
        1.23 -> decimal { integer_part: "일" fractional_part: "이삼" }
        -0.5 -> decimal { negative: "마이너스" integer_part: "영" fractional_part: "오" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_before_decimal = cardinal.graph
        cardinal_after_decimal = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))

        DOUBLE_QUOTE = '"'
        
        graph_integer = pynutil.insert(f'integer_part: {DOUBLE_QUOTE}') + cardinal_before_decimal + pynutil.insert(DOUBLE_QUOTE)
        graph_fractional = (
            pynutil.insert(f'fractional_part: {DOUBLE_QUOTE}')
            + pynini.closure(cardinal_after_decimal | zero, 1)
            + pynutil.insert(DOUBLE_QUOTE)
        )

        # Decimal without a sign (e.g., 2.5)
        graph_decimal_no_sign = graph_integer + pynutil.delete('.') + pynutil.insert(NEMO_SPACE) + graph_fractional

        # Negative sign handling (e.g., -2.5 or 마이너스2.5)
        graph_with_negative = (
            pynutil.insert(f'negative: {DOUBLE_QUOTE}')
            + (pynini.cross("-", "마이너스") | pynini.accep("마이너스"))
            + pynutil.insert(DOUBLE_QUOTE)
        )

        graph_decimal = graph_decimal_no_sign | (graph_with_negative + pynutil.insert(NEMO_SPACE) + graph_decimal_no_sign)

        # For internal use without tokens
        self.just_decimal = graph_decimal_no_sign.optimize()

        # Final graph with tokens
        final_graph = self.add_tokens(graph_decimal)
        self.fst = final_graph.optimize()
