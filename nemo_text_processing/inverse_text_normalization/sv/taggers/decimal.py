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
from nemo_text_processing.text_normalization.de.taggers.decimal import get_quantity, quantities
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. minus elf komma zwei null null sechs billionen -> decimal { negative: "true" integer_part: "11"  fractional_part: "2006" quantity: "billionen" }
        e.g. eine billion -> decimal { integer_part: "1" quantity: "billion" }
    Args:
        itn_cardinal_tagger: ITN Cardinal tagger
        tn_decimal_tagger: TN decimal tagger
    """

    def __init__(self, itn_cardinal_tagger: GraphFst, tn_decimal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        self.graph = tn_decimal_tagger.graph.invert().optimize()

        delete_point = pynutil.delete(" komma")

        allow_spelling = pynini.cdrewrite(pynini.cross("eine ", "eins ") + quantities, "[BOS]", "[EOS]", NEMO_SIGMA)

        graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        graph_integer = (
            pynutil.insert("integer_part: \"") + itn_cardinal_tagger.graph_no_exception + pynutil.insert("\"")
        )
        final_graph_wo_sign = graph_integer + delete_point + pynini.accep(" ") + graph_fractional

        self.final_graph_wo_negative = (
            allow_spelling
            @ (
                final_graph_wo_sign
                | get_quantity(
                    final_graph_wo_sign, itn_cardinal_tagger.graph_hundred_component_at_least_one_none_zero_digit
                )
            ).optimize()
        )

        final_graph = itn_cardinal_tagger.optional_minus_graph + self.final_graph_wo_negative
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
