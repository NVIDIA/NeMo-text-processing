# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -१२kg -> measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" }
        -१२.२kg -> measure { decimal { negative: "true"  integer_part: "बारह"  fractional_part: "दो"} units: "किलोग्राम" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal = cardinal.fst
        decimal = decimal.fst

        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))
        unit = pynutil.insert("units: \"") + unit_graph + pynutil.insert("\"")

        graph = (decimal | cardinal) + delete_space + insert_space + unit
    
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph

if __name__ == '__main__':
    from decimal import DecimalFst
    from cardinal import CardinalFst
    from nemo_text_processing.text_normalization.hi.utils import apply_fst

    cardinal = CardinalFst()
    decimal = DecimalFst(cardinal=cardinal)
    measure = MeasureFst(cardinal=cardinal, decimal=decimal)
    input_text = "१५००० kg"
    input_text = "१५० kg"
    apply_fst(input_text, measure.fst)  