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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. 마이너스 이십삼 -> cardinal { integer: "23" negative: "-" } }

    Args:
        input_case: accepting Korean input.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))

        ten = pynutil.delete("십")
        ten_alt = pynini.cross("십", "1")
        ### Responsible for second digit of two digit number. ex) 20's 2
        graph_ten_component = pynini.union((graph_digit + ten) | ten_alt, pynutil.insert("0"))
        ### Responsible for the first digit of number. ex) 1,2,3,4,5,,,
        graph_ten_component += graph_digit | pynutil.insert("0")

        hundred = pynutil.delete("백")
        hundred_alt = pynini.cross("백", "1")
        graph_hundred_component = pynini.union(((graph_digit + hundred) | hundred_alt), pynutil.insert("0"))
        graph_hundred_component += graph_ten_component

        thousand = pynutil.delete("천")
        thousand_alt = pynini.cross("천", "1")
        graph_thousand_component = pynini.union(((graph_digit + thousand) | thousand_alt), pynutil.insert("0"))
        graph_thousand_component += graph_hundred_component

        tenthousand = pynutil.delete("만")
        tenthousand_alt = pynini.cross("만", "1")
        ### "만" can express next four digits of numbers until the next unit "억", so insert "0000" to allocate four digit worth of space
        ### From "만", keep adding four digits and graph_thousand_component(0000-9999), because Korean units increase every four digits
        graph_tenthousand_component = pynini.union(
            ((graph_thousand_component + tenthousand) | tenthousand_alt), pynutil.insert("0000")
        )
        graph_tenthousand_component += graph_thousand_component

        hundredmillion = pynutil.delete("억")
        hundredmillion_alt = pynini.cross("억", "1")
        graph_hundredmillion_component = pynini.union(
            ((graph_thousand_component + hundredmillion) | hundredmillion_alt), pynutil.insert("0000")
        )
        graph_hundredmillion_component += graph_tenthousand_component

        trillion = pynutil.delete("조")
        trillion_alt = pynini.cross("조", "1")
        graph_trillion_component = pynini.union(
            ((graph_thousand_component + trillion) | trillion_alt), pynutil.insert("0000")
        )
        graph_trillion_component += graph_hundredmillion_component

        tenquadrillion = pynutil.delete("경")
        tenquadrillion_alt = pynini.cross("경", "1")
        graph_tenquadrillion_component = pynini.union(
            ((graph_thousand_component + tenquadrillion) | tenquadrillion_alt), pynutil.insert("0000")
        )
        graph_tenquadrillion_component += graph_trillion_component

        graph = pynini.union(
            ### From biggest unit to smallest, everything is included
            graph_tenquadrillion_component
            | graph_zero
        )

        leading_zero = (
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
        )
        graph = (graph @ leading_zero) | graph_zero

        self.just_cardinals = graph

        negative_sign = pynini.closure(
            (pynini.cross("마이너스", 'negative: "-"') | pynini.cross("-", 'negative: "-"')) + delete_space, 0, 1
        )

        final_graph = (
            negative_sign + pynutil.insert(" ") + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        ) | (pynutil.insert("integer: \"") + graph + pynutil.insert("\""))

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
