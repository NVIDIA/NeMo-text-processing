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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.sv.taggers.decimal import get_quantity


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. minus elva komma två nulla nulla sex biljoner -> decimal { negative: "true" integer_part: "11"  fractional_part: "2006" quantity: "biljoner" }
        e.g. en biljon -> decimal { integer_part: "1" quantity: "biljon" }
    Args:
        itn_cardinal_tagger: ITN Cardinal tagger
        tn_decimal_tagger: TN decimal tagger
    """

    def __init__(self, itn_cardinal_tagger: GraphFst, tn_decimal_tagger: GraphFst):
        super().__init__(name="decimal", kind="classify")

        self.graph = tn_decimal_tagger.graph_itn
        self.graph = self.graph @ pynini.cdrewrite(pynini.cross(" ", ""), "", "", NEMO_SIGMA)

        delete_point = pynutil.delete(" komma")

        graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        hundreds = itn_cardinal_tagger.graph_hundred_component_at_least_one_non_zero_digit
        hundreds = (pynini.project(hundreds, "input") - "en" - "ett") @ hundreds
        hundreds_no_one = hundreds
        hundreds |= pynini.cross("en", "1")
        hundreds |= pynini.cross("ett", "1")
        graph_integer = pynutil.insert("integer_part: \"") + hundreds + pynutil.insert("\"")
        self.graph_integer = graph_integer
        final_graph_wo_sign = graph_integer + delete_point + pynini.accep(" ") + graph_fractional
        self.final_graph_wo_sign = final_graph_wo_sign

        self.final_graph_wo_negative = (
            final_graph_wo_sign | get_quantity(final_graph_wo_sign, None, hundreds_no_one, None, False, True,)
        ).optimize()

        optional_minus_graph = pynini.closure(pynini.cross("minus ", "negative: \"true\" "), 0, 1)
        final_graph = optional_minus_graph + self.final_graph_wo_negative
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
