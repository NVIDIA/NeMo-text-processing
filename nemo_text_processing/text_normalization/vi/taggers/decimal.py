# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese decimal numbers, e.g.
        -12,5 tỷ -> decimal { negative: "true" integer_part: "mười hai" fractional_part: "năm" quantity: "tỷ" }
        818,303 -> decimal { integer_part: "tám trăm mười tám" fractional_part: "ba không ba" }
        0,2 triệu -> decimal { integer_part: "không" fractional_part: "hai" quantity: "triệu" }

    Args:
        cardinal: CardinalFst instance for processing integer parts
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_with_and
        self.graph = cardinal.single_digits_graph.optimize()
        if not deterministic:
            self.graph = self.graph | cardinal_graph

        single_digit_map = pynini.union(
            *[pynini.cross(k, v) for k, v in load_labels(get_abs_path("data/numbers/digit.tsv"))],
            *[pynini.cross(k, v) for k, v in load_labels(get_abs_path("data/numbers/zero.tsv"))]
        )

        quantity_units = pynini.union(*[v for _, v in load_labels(get_abs_path("data/numbers/magnitudes.tsv"))])

        integer_part = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        fractional_part = (
            pynutil.insert("fractional_part: \"")
            + (single_digit_map + pynini.closure(pynutil.insert(" ") + single_digit_map))
            + pynutil.insert("\"")
        )

        decimal_pattern = (
            (integer_part + pynutil.insert(" ")).ques + pynutil.delete(",") + pynutil.insert(" ") + fractional_part
        )

        quantity_suffix = (
            pynutil.delete(" ").ques + pynutil.insert(" quantity: \"") + quantity_units + pynutil.insert("\"")
        )

        decimal_with_quantity = decimal_pattern + quantity_suffix
        cardinal_with_quantity = integer_part + quantity_suffix

        negative = (pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")).ques
        final_graph = negative + pynini.union(decimal_pattern, decimal_with_quantity, cardinal_with_quantity)

        self.fst = self.add_tokens(final_graph).optimize()
