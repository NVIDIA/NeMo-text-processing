# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


def get_quantity(decimal):
    suffix = pynini.union("만", "억", "조", "경")
    numbers = decimal
    res = numbers + pynutil.insert(' quantity: "') + suffix + pynutil.insert('"')

    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. 일점오 -> decimnl { integer_part: "1" fractional_part: "5" }
        e.g. 일점오만 -> decimal { integer_part: "1" fractional_part: "5" quantity: "만" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinals = cardinal.just_cardinals
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        decimal_part = pynini.closure(graph_zero | graph_digit)

        decimal_point = pynutil.delete("점")
        integer_part = pynutil.insert("integer_part: \"") + cardinals + pynutil.insert("\"")
        fractional_part = pynutil.insert("fractional_part: \"") + decimal_part + pynutil.insert("\"")

        graph_decimal_regular = (
            integer_part + decimal_point + pynutil.insert(NEMO_SPACE) + fractional_part
        )  # Regular decimal like 1.5
        graph_deicimal_larger = get_quantity(
            graph_decimal_regular
        )  # If decimal is used to express big numbers like  15000 -> "1.5만"

        self.decimal = graph_decimal_regular | graph_deicimal_larger
        self.just_decimal = cardinals + pynini.cross("점", ".") + decimal_part

        graph_sign = (
            pynutil.insert("negative: \"") + (pynini.cross("마이너스", "-") | pynini.accep("-")) + pynutil.insert("\"")
        )

        final_graph = (
            (graph_sign + pynutil.insert(" ") + graph_decimal_regular)
            | (graph_sign + pynutil.insert(" ") + graph_deicimal_larger)
            | graph_decimal_regular
            | graph_deicimal_larger
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
