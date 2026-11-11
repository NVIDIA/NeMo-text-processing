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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal numbers in Tamil, e.g.
        decimal { integer_part: "௧௦௧" fractional_part: "௬" } -> ௧௦௧.௬
        decimal { negative: "true" integer_part: "௧" fractional_part: "௫" } -> -௧.௫
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = optional_sign + integer + delete_space + fractional

        self.fst = self.delete_tokens(graph).optimize()
