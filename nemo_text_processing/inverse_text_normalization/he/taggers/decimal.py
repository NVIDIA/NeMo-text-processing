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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import MINUS, GraphFst, delete_and
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    delete_extra_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)


def get_quantity(decimal: "pynini.FstLike", cardinal_up_to_hundred: "pynini.FstLike") -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral in Hebrew,

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
    )

    suffix_labels = ["מיליון", "מיליארד"]
    suffix = pynini.union(*suffix_labels).optimize()

    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + delete_extra_space
        + pynutil.insert('quantity: "')
        + suffix
        + pynutil.insert('"')
    )
    res |= decimal + delete_extra_space + pynutil.insert('quantity: "') + (suffix | "אלף") + pynutil.insert('"')
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal in Hebrew
        e.g. עשרים ושלוש וחצי -> decimal { integer_part: "23" fractional_part: "5" }
        e.g. אחד נקודה שלוש -> decimal { integer_part: "1"  fractional_part: "3" }
        e.g. ארבע נקודה חמש מיליון -> decimal { integer_part: "4"  fractional_part: "5" quantity: "מיליון" }
        e.g. מינוס ארבע מאות נקודה שלוש שתיים שלוש -> decimal { negative: "true" integer_part: "400"  fractional_part: "323" }
        e.g. אפס נקודה שלושים ושלוש -> decimal { integer_part: "0"  fractional_part: "33" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        prefix_graph = pynini.string_file(get_abs_path("data/prefix.tsv"))
        optional_prefix_graph = pynini.closure(
            pynutil.insert('morphosyntactic_features: "') + prefix_graph + pynutil.insert('"') + insert_space,
            0,
            1,
        )

        # all cardinals
        cardinal_graph = cardinal.graph_no_exception

        # all fractions
        fractions = pynini.string_file(get_abs_path("data/numbers/decimal_fractions.tsv"))
        fractions_graph = delete_zero_or_one_space + delete_and + fractions
        fractions_graph = pynutil.insert('fractional_part: "') + fractions_graph + pynutil.insert('"')

        # identify decimals that can be understood as time, don't convert them to avoid ambiguity
        viable_minutes_exception = pynini.string_file(get_abs_path("data/decimals/minutes_exception.tsv"))
        fractions_wo_minutes = (pynini.project(fractions, "input") - viable_minutes_exception.arcsort()) @ fractions
        fractions_wo_minutes = delete_zero_or_one_space + delete_and + fractions_wo_minutes
        fractions_wo_minutes = pynutil.insert('fractional_part: "') + fractions_wo_minutes + pynutil.insert('"')

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_decimal |= cardinal.graph_two_digit
        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        point = pynutil.delete("נקודה")

        graph_negative = pynutil.insert("negative: ") + pynini.cross(MINUS, '"true"') + delete_extra_space
        optional_graph_negative = pynini.closure(
            graph_negative,
            0,
            1,
        )

        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        graph_fractional = pynutil.insert('fractional_part: "') + graph_decimal + pynutil.insert('"')

        # integer could be an hour, but minutes cannot: convert to decimal
        viable_hour_unviable_minutes = graph_integer + delete_extra_space + fractions_wo_minutes

        # integer cannot be an hour, but minutes can: convert to decimal
        unviable_hour_viable_minutes = (
            pynutil.insert('integer_part: "')
            + cardinal.graph_wo_viable_hours
            + pynutil.insert('"')
            + delete_extra_space
            + fractions_graph
        )

        # minus sign followed by ambiguous decimal: convert to decimal, there is no negative time
        negative_viable_time = graph_negative + graph_integer + delete_extra_space + fractions_graph

        # all decimals with fractions, not excluding anything (used in other FSTs)
        all_decimals_wo_point = graph_integer + delete_extra_space + fractions_graph

        # only cases with fractional part that cannot be interpreted as time
        graph_wo_point = viable_hour_unviable_minutes | unviable_hour_viable_minutes | negative_viable_time

        # all decimals with the word "point"
        graph_w_point = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )

        final_graph_wo_sign = graph_w_point | graph_wo_point
        self.final_graph_wo_sign = graph_w_point | all_decimals_wo_point
        final_graph = optional_prefix_graph + optional_graph_negative + final_graph_wo_sign

        quantity_graph = get_quantity(self.final_graph_wo_sign, cardinal.graph_hundred)
        final_graph |= optional_prefix_graph + optional_graph_negative + quantity_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
