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


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying a generic 3-4-4 telephone number.
        e.g. 공일공에 일이삼사에 오육칠팔 -> telephone { number_part: "010-1234-5678" }

    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_zero_alt = pynini.cross("공", "0")
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))

        digit = graph_digit | graph_zero | graph_zero_alt

        separator = pynini.cross(pynini.union(NEMO_SPACE, "에"), "-")

        digit2 = digit**2
        digit3 = digit**3
        digit4 = digit**4

        optional_separator = pynini.closure(separator, 0, 1)

        phone_number_graph = (
            pynutil.insert('number_part: "')
            + pynini.union(digit2, digit3)
            + optional_separator
            + digit4
            + optional_separator
            + digit4
            + pynutil.insert('"')
        )

        graph = phone_number_graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
