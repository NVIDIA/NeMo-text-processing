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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing Korean measure tokens into surface text.
        measure { cardinal { integer: "<...>" } units: "<...>" }

    Converts tokens like:
        measure { cardinal { integer: "이" } units: "킬로그램" }
        measure { fraction { numerator: "이" denominator: "삼" } units: "킬로미터" }

    into surface text:
        "이 킬로그램", "삼분의 이 킬로미터"

    Args:
        decimal, cardinal, fraction: Sub-verbalizers handling number types.
        deterministic: If True, outputs a single normalized form.
    """

    def __init__(
        self,
        decimal: GraphFst = None,
        cardinal: GraphFst = None,
        fraction: GraphFst = None,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        # Combine all numeric verbalizers
        graph_cardinal = cardinal.fst
        graph_decimal = decimal.fst
        graph_fraction = fraction.fst

        number_block = graph_cardinal | graph_decimal | graph_fraction

        # Extract and output unit string
        units = (
            delete_space
            + pynutil.delete("units:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Normal form: <number> <unit>
        main = number_block + insert_space + units

        # preserve_order form: <unit> <number>
        preserve_order = delete_space + pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true")
        alt = units + insert_space + number_block + pynini.closure(preserve_order)

        graph = main | alt

        # Merge and clean tokens
        self.fst = self.delete_tokens(graph).optimize()
