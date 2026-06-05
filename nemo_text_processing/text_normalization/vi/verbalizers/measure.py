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

from nemo_text_processing.text_normalization.vi.graph_utils import (
    GraphFst,
    delete_preserve_order,
    delete_space,
    extract_field,
    extract_wrapper_content,
    insert_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure for Vietnamese, e.g.
        measure { negative: "true" cardinal { integer: "mười hai" } units: "ki lô gam" } -> âm mười hai ki lô gam
        measure { decimal { integer_part: "mười hai" fractional_part: "năm" } units: "ki lô gam" } -> mười hai phẩy năm ki lô gam
        measure { cardinal { integer: "một" } units: "ki lô gam" } -> một ki lô gam

    Args:
        decimal: DecimalFst verbalizer
        cardinal: CardinalFst verbalizer
        fraction: FractionFst verbalizer
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        # Extract components
        unit = extract_field("units")

        # Handle negative sign - Vietnamese uses "âm" for negative numbers
        optional_negative = pynini.closure(pynini.cross("negative: \"true\"", "âm ") + delete_space, 0, 1)
        if not deterministic:
            # Alternative ways to say negative in Vietnamese
            optional_negative |= pynini.closure(pynini.cross("negative: \"true\"", "trừ ") + delete_space, 0, 1)

        # Combine all number types into single graph
        number_graph = (
            extract_wrapper_content("decimal", decimal.numbers)
            | extract_wrapper_content("cardinal", cardinal.numbers)
            | extract_wrapper_content("fraction", fraction.numbers)
        )

        # Main pattern: [negative] number + space + unit (most common case)
        graph = optional_negative + number_graph + delete_space + insert_space + unit

        # Handle preserve_order: [negative] unit + space + number
        graph |= optional_negative + unit + delete_space + insert_space + number_graph + delete_preserve_order

        self.fst = self.delete_tokens(graph).optimize()
