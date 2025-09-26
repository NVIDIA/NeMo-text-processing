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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
)
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. 오만 삼천원 -> money { integer_part: "53000" currency: "원" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinals = cardinal.just_cardinals
        currency = pynini.string_file(get_abs_path("data/currency.tsv"))

        graph_unit = pynutil.insert('currency: "') + currency + pynutil.insert('"')

        # Main graph for integer money amounts
        # Structure: <number> + <optional space> + <currency>
        graph_integer = (
            pynutil.insert('integer_part: "')
            + cardinals
            + pynutil.insert('"')
            + delete_extra_space  # Handles optional spacing
            + graph_unit
        )

        final_graph = self.add_tokens(graph_integer)
        self.fst = final_graph.optimize()
