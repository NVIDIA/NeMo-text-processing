# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing Japanese measure tokens.

    Examples:
        measure { cardinal { integer: "五" } units: "キロ" preserve_order: true } -> 五キロ
        measure { cardinal { integer: "時速六十" } units: "キロ" preserve_order: true } -> 時速六十キロ
        measure { cardinal { integer: "秒速五十" } units: "メートル" preserve_order: true } -> 秒速五十メートル
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        unit = delete_space + pynutil.delete('units: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        number = cardinal.fst | decimal.fst | fraction.fst

        graph = number + unit + delete_preserve_order

        self.fst = self.delete_tokens(graph.optimize()).optimize()
