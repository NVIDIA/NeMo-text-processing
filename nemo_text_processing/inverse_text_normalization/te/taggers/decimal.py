# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    MINUS,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu decimals.

    Examples:
        ఒకటి దశాంశం రెండు మూడు
        -> decimal { integer_part: "౧" fractional_part: "౨౩" }

        దశాంశం ఐదు
        -> decimal { fractional_part: "౫" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_people = pynini.string_file(get_abs_path("data/numbers/digit_people.tsv"))
        graph_point = pynini.string_file(get_abs_path("data/decimal/point.tsv"))

        point = pynutil.delete(graph_point)
        people_input = pynini.project(graph_people, "input").optimize()
        cardinal_input = pynini.project(cardinal.graph_no_exception, "input").optimize()
        non_people_input = pynini.difference(cardinal_input, people_input).optimize()

        cardinal_graph = (non_people_input @ cardinal.graph_no_exception).optimize()

        graph_decimal_digit = graph_digit | graph_zero
        graph_decimal_digits = pynini.closure(graph_decimal_digit + delete_space) + graph_decimal_digit

        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')

        graph_fractional = pynutil.insert('fractional_part: "') + graph_decimal_digits + pynutil.insert('"')
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, '"-"') + NEMO_SPACE,
            0,
            1,
        )

        final_graph = optional_minus_graph + (
            pynini.closure(
                graph_integer + delete_extra_space,
                0,
                1,
            )
            + point
            + delete_extra_space
            + graph_fractional
        )

        self.fst = self.add_tokens(final_graph).optimize()
