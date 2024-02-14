# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. tokens { 'fraction { integer_part: "مئة وخمسة" numerator: "ثلاثون" denominator: "سبعة وستون" }' } ->
        مئة وخمسة و ثلاثون على سبعة وستون

    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "سالب ") + pynutil.delete(" "), 0, 1)

        # create unions for special cases
        denominator_singular = pynini.union("نصف", "ثلث", "ربع", "خمس", "سدس", "سبع", "ثمن", "تسع", "عشر")
        denominator_dual = pynini.union(
            "نصفين", "ثلثين", "ربعين", "خمسين", "سدسين", "سبعين", "ثمنين", "تسعين", "عشرين"
        )
        denominator_plural = pynini.union("أخماس", "أرباع", "أثلاث", "أسداس", "أسباع", "أثمان", "أتساع", "أعشار")
        numerator_three_to_ten = pynini.union("خمسة", "سبعة", "عشرة", "ثلاثة", "أربعة", "ستة", "ثمانية", "تسعة")

        # filter cases when denominator_singular
        graph_denominator_singular = (
            pynutil.delete("denominator: \"")
            + denominator_singular @ pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )

        # filter cases when denominator_dual
        graph_denominator_dual = (
            pynutil.delete("denominator: \"")
            + denominator_dual @ pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        # filter cases when denominator_plural
        graph_denominator_plural = (
            pynutil.delete("denominator: \"")
            + denominator_plural @ pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )
        # integer part

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")

        # verbalize from integer and one over half -->  integer and half e.g واحد ونصف
        numerator_one = pynutil.delete("numerator: \"واحد\"") + pynutil.delete(" ") + graph_denominator_singular
        # verbalize from integer and two over half(dual) -->  integer and half(dual) e.g. واحد وثلثين
        numerator_two = pynutil.delete("numerator: \"اثنان\"") + pynutil.delete(" ") + graph_denominator_dual
        # verbalize from integer and three over thirds(plural) -->  integer and  three thirds(plural) e.g.  واحد وثلاثة أرباع
        numerator_three_to_ten = (
            pynutil.delete("numerator: \"")
            + numerator_three_to_ten @ pynini.closure(NEMO_NOT_QUOTE)
            + insert_space
            + pynutil.delete("\"")
            + pynutil.delete(" ")
            + graph_denominator_plural
        )

        # rest of numbers
        denominator_rest = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        numerator_rest = pynutil.delete("numerator: \"") + (pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" "))

        numerators = numerator_rest + insert_space + pynutil.insert("على ") + denominator_rest

        fraction = (
            numerator_one | numerator_three_to_ten | numerator_two | pynutil.add_weight(numerators, 0.001)
        )  # apply exceptions first then the rest

        conjunction = pynutil.insert("و ")

        integer = pynini.closure(integer + insert_space + conjunction, 0, 1)

        self.graph = optional_sign + integer + fraction

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
