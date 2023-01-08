# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst, delete_extra_space
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. "минус три целых две десятых" -> decimal { negative: "true" integer_part: "3," fractional_part: "2" }

    Args:
        tn_decimal: Text normalization Decimal graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_decimal, deterministic: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("минус", "\"true\"") + delete_extra_space, 0, 1
        )

        graph_fractional_part = pynini.invert(tn_decimal.graph_fractional).optimize()
        graph_integer_part = pynini.invert(tn_decimal.integer_part).optimize()
        optional_graph_quantity = pynini.invert(tn_decimal.optional_quantity).optimize()

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional_part + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + graph_integer_part + pynutil.insert("\"")
        optional_graph_quantity = pynutil.insert("quantity: \"") + optional_graph_quantity + pynutil.insert("\"")
        optional_graph_quantity = pynini.closure(pynini.accep(NEMO_SPACE) + optional_graph_quantity, 0, 1)

        self.final_graph_wo_sign = (
            graph_integer + pynini.accep(NEMO_SPACE) + graph_fractional + optional_graph_quantity
        )
        final_graph = optional_graph_negative + self.final_graph_wo_sign

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
