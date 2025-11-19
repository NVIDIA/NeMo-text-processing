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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    NEMO_SPACE
)
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. 십이 킬로그램 -> measure { cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.just_cardinals

        graph_unit = pynini.string_file(get_abs_path("data/measure_units.tsv"))

        delete_any_space = pynini.closure(pynutil.delete(NEMO_SPACE))
        #Negative sign
        negative_word = pynini.union("마이너스", "영하")
        graph_negative = pynini.cross(negative_word, 'negative: "true"') + delete_extra_space
        optional_graph_negative = pynini.closure(graph_negative, 0, 1)
        #Graphing measurement units
        unit_singular = convert_space(graph_unit)
        #For units that has "/", like km/h 
        unit_per = (
            unit_singular
            + delete_any_space
            + pynini.cross(pynini.union("퍼", "당"), "/")
            + delete_any_space 
            + unit_singular
        )
        
        graph_unit_final = (
            pynutil.insert('units: "')
            + (unit_singular | unit_per)
            + pynutil.insert('"')
        )

        #Graphing decimal
        graph_digit_tsv = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.cross("영", "0") | pynini.cross("공", "0")
        decimal_fractional_part = pynini.closure(graph_digit_tsv | graph_zero, 1)

        graph_decimal = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
            + delete_any_space
            + pynini.cross("점", " ")
            + delete_any_space
            + pynutil.insert('fractional_part: "')
            + decimal_fractional_part
            + pynutil.insert('"')
        )

        #Graphing fraction
        graph_fraction = (
            pynutil.insert("fraction { ")
            + pynutil.insert('denominator: "')
            + cardinal_graph
            + pynutil.insert('"')
            + delete_any_space
            + pynutil.delete("분의")
            + delete_any_space
            + pynutil.insert(' numerator: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
        )

        final_graph_cardinal = (
            delete_any_space
            + pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + pynutil.insert(" ")
            + delete_any_space
            + graph_unit_final
        )

        final_graph_decimal = (
            delete_any_space 
            + pynutil.insert("decimal { ")
            + optional_graph_negative
            + graph_decimal
            + pynutil.insert(" }")
            + pynutil.insert(" ")
            + delete_any_space
            + graph_unit_final
        )
        
        final_graph_fraction = (
            delete_any_space
            + graph_fraction
            + pynutil.insert(" ")
            + delete_any_space
            + graph_unit_final
        )

        final_graph = final_graph_cardinal | final_graph_decimal | final_graph_fraction
        
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()