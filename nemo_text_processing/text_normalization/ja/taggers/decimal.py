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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        0.5 -> decimal { integer_part: "零" fractional_part: "五" }
        -0.5万 -> decimal { negative: "マイナス" integer_part: "零" fractional_part: "五" quantity: "万"}
        
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_before_decimal = cardinal.just_cardinals
        cardinal_after_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        graph_integer = pynutil.insert('integer_part: \"') + cardinal_before_decimal + pynutil.insert("\"")
        graph_fraction = (
            pynutil.insert("fractional_part: \"")
            + pynini.closure((cardinal_after_decimal | zero), 1)
            + pynutil.insert("\"")
        )
        graph_decimal_no_sign = graph_integer + pynutil.delete('.') + pynutil.insert(" ") + graph_fraction

        graph_optional_sign = (
            pynutil.insert("negative: \"") + (pynini.cross("-", "マイナス") | pynini.accep("マイナス")) + pynutil.insert("\"")
        )

        graph_decimal = graph_decimal_no_sign | (graph_optional_sign + pynutil.insert(" ") + graph_decimal_no_sign)

        self.just_decimal = graph_decimal_no_sign.optimize()

        final_graph = self.add_tokens(graph_decimal)
        self.fst = final_graph.optimize()
