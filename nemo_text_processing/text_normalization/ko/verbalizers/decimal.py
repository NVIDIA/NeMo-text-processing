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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst


class DecimalFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        # Extract integer part
        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Extract fractional part and prepend "점"
        fractional_part = (
            pynutil.delete('fractional_part: "')
            + pynutil.insert("점")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Verbalize decimal number without sign
        decimal_positive = integer_part + pynutil.delete(" ") + fractional_part

        # Handle negative sign
        negative_sign = (
            pynutil.delete('negative: "') + pynini.accep("마이너스") + pynutil.delete('"') + pynutil.delete(" ")
        )

        # Combine positive and negative cases
        decimal = decimal_positive | (negative_sign + pynutil.insert(" ") + decimal_positive)

        delete_tokens = self.delete_tokens(decimal)
        self.fst = delete_tokens.optimize()
