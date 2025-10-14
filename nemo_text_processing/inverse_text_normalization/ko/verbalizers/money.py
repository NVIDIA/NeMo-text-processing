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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_CHAR, GraphFst, delete_space, NEMO_SPACE


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. 오만 삼천원 -> money { integer_part: "53000" currency: "원" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_CHAR - NEMO_SPACE, 1)
            + pynutil.delete('"')
        )
        
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_CHAR - NEMO_SPACE, 1)
            + pynutil.delete('"')
        )

        optional_space = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1).optimize()

        graph = unit + optional_space + integer
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()