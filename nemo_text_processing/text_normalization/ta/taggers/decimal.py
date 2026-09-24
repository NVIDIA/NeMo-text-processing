# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.ta.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals
        e.g. "1.5" -> decimal { integer_part: "ஒன்று" fractional_part: "புள்ளி ஐந்து" }
        e.g. "-2.67" -> decimal { negative: "true" integer_part: "இரண்டு" fractional_part: "புள்ளி ஆறு ஏழு" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        frac_digit = (digit | zero).optimize()
        cg = cardinal.final_graph.optimize()

        frac = (
            pynutil.insert("fractional_part: \"")
            + pynutil.insert("புள்ளி ")
            + (frac_digit + pynini.closure(insert_space + frac_digit)).optimize()
            + pynutil.insert("\"")
        )
        inte = pynutil.insert("integer_part: \"") + cg + pynutil.insert("\"")
        core = inte + pynutil.delete(".") + insert_space + frac

        neg = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1)

        self.fst = self.add_tokens(neg + core).optimize()
