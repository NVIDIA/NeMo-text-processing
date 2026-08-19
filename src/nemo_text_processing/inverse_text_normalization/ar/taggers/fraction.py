# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. واحد و نصف -> tokens { integer_part: "1" numerator: "1" denominator: "2" }

    Args:
        tn_cardinal: TN cardinal tagger

    """

    def __init__(self, tn_cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("سالب", "\"true\" "), 0, 1
        )
        cardinal_graph = pynini.invert(tn_cardinal.cardinal_numbers).optimize()

        # create unions for special cases
        denominator_singular = pynini.union("نصف", "ثلث", "ربع", "خمس", "سدس", "سبع", "ثمن", "تسع", "عشر")
        denominator_dual = pynini.union(
            "نصفين", "ثلثين", "ربعين", "خمسين", "سدسين", "سبعين", "ثمنين", "تسعين", "عشرين"
        )
        denominator_plural = pynini.union("أخماس", "أرباع", "أثلاث", "أسداس", "أسباع", "أثمان", "أتساع", "أعشار")
        numerator_three_to_ten = pynini.union("خمسة", "سبعة", "عشرة", "ثلاثة", "أربعة", "ستة", "ثمانية", "تسعة")

        # data files
        graph_ones = pynini.string_file(get_abs_path("data/number/fraction_singular.tsv")).invert().optimize()
        graph_dual = pynini.string_file(get_abs_path("data/number/fraction_dual.tsv")).invert().optimize()
        graph_plural = pynini.string_file(get_abs_path("data/number/fraction_plural.tsv")).invert().optimize()

        # cases when denominator_singular
        graph_denominator_singular = (
            pynutil.insert("denominator: \"") + denominator_singular @ graph_ones + pynutil.insert("\"")
        )

        # cases when denominator_dual
        graph_denominator_dual = (
            pynutil.insert("denominator: \"") + denominator_dual @ graph_dual + pynutil.insert("\"")
        )
        # cases when denominator_plural
        graph_denominator_plural = (
            pynutil.insert("denominator: \"")
            + delete_zero_or_one_space
            + denominator_plural @ graph_plural
            + pynutil.insert("\"")
        )

        denominator_rest = pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")
        numerator_rest = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\" ")

        # integer part
        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"") + insert_space

        # e.g نصف
        numerator_one = pynutil.insert("numerator: \"1\"") + insert_space + graph_denominator_singular
        # e.g.  ثلثين
        numerator_two = pynutil.insert("numerator: \"2\"") + insert_space + graph_denominator_dual
        # e.g. ثلاثة أرباع
        numerator_three_to_ten = (
            pynutil.insert("numerator: \"")
            + numerator_three_to_ten @ cardinal_graph
            + pynutil.insert("\"")
            + insert_space
            + graph_denominator_plural
        )
        # e.g. اثنا عشر على أربعة وعشرون
        numerators = (
            numerator_rest
            + delete_zero_or_one_space
            + pynutil.delete("على")
            + delete_zero_or_one_space
            + denominator_rest
        )

        fraction = (
            numerator_one | numerator_three_to_ten | numerator_two | pynutil.add_weight(numerators, 0.001)
        )  # apply exceptions first then the rest

        conjunction = pynutil.delete("و")

        integer = pynini.closure(integer + delete_zero_or_one_space + conjunction + delete_zero_or_one_space, 0, 1)

        graph = optional_graph_negative + integer + fraction

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
