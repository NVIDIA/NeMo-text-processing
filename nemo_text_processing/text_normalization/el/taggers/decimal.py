# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, insert_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals in Greek, e.g.
        3,5 -> decimal { integer_part: "τρία" fractional_part: "πέντε" }
        -0,05 -> decimal { negative: "true" integer_part: "μηδέν" fractional_part: "μηδέν πέντε" }

    The decimal separator is the comma. The integer part is rendered as a cardinal (neuter),
    while the fractional part is read digit by digit so that leading zeros are preserved.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_no_tokens

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        digit_all = graph_digit | pynini.cross("0", "μηδέν")
        fractional = digit_all + pynini.closure(insert_space + digit_all)

        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_fractional = pynutil.insert("fractional_part: \"") + fractional + pynutil.insert("\"")

        final_graph = (
            optional_negative + graph_integer + pynutil.delete(",") + insert_space + graph_fractional
        )
        self.final_graph = final_graph
        self.fst = self.add_tokens(final_graph).optimize()
