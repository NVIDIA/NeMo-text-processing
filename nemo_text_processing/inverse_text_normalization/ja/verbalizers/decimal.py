# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal
        e.g.  decimal { integer_part: "1" fractional_part: "5" } -> 1.5
        e.g.  decimal { integer_part: "1" fractional_part: "5" quantity: "万" } -> 1.5万
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        decimal_point = pynutil.insert(".")
        integer_component = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        fractional_component = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        )
        quantity_component = pynutil.delete("quantity: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_decimal = integer_component + decimal_point + pynutil.delete(" ") + fractional_component
        graph_decimal_larger = (
            integer_component
            + decimal_point
            + pynutil.delete(" ")
            + fractional_component
            + pynutil.delete(" ")
            + quantity_component
        )

        graph_sign = pynutil.delete("negative: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph = (
            graph_decimal
            | graph_decimal_larger
            | (graph_sign + pynutil.delete(" ") + graph_decimal)
            | (graph_sign + pynutil.delete(" ") + graph_decimal_larger)
        )

        final_graph = self.delete_tokens(graph)
        self.fst = final_graph.optimize()
