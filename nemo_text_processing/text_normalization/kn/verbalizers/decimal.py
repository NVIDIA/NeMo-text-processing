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

from nemo_text_processing.text_normalization.kn.graph_utils import NEMO_NOT_QUOTE, Decimal_word, GraphFst, Minus_word


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "ಹನ್ನೆರಡು" fractional_part: "ಐದು ಆರು" } -> ಮೈನಸ್ ಹನ್ನೆರಡು ದಶಮಾಂಶ ಐದು ಆರು

    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        self.optional_sign = pynini.closure(pynini.cross('negative: "true" ', Minus_word + " "), 0, 1)
        self.integer = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        self.fractional = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        graph_integer_and_fraction = (
            self.integer + pynutil.delete(" ") + pynutil.insert(" " + Decimal_word + " ") + self.fractional
        )

        graph_integer_only = self.integer
        graph_fraction_only = pynutil.insert(Decimal_word + " ") + self.fractional
        graph = self.optional_sign + (graph_integer_and_fraction | graph_integer_only | graph_fraction_only)
        self.fst = self.delete_tokens(graph).optimize()
