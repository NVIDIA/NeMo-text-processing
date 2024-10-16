# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.sv.utils import get_abs_path


def get_quantity(
    decimal: 'pynini.FstLike',
    decimal_ett: 'pynini.FstLike',
    cardinal_up_to_thousand: 'pynini.FstLike',
    cardinal_up_to_thousand_ett: 'pynini.FstLike',
    include_abbr: bool,
    itn: bool = False,
) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 miljon -> integer_part: "en" quantity: "miljon"
    e.g. 1,5 miljoner -> integer_part: "en" fractional_part: "fem" quantity: "miljoner"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    quantities = pynini.string_file(get_abs_path("data/numbers/millions.tsv"))
    quantities_abbr = pynini.string_file(get_abs_path("data/numbers/millions_abbr.tsv"))

    quantities_pl = quantities + "er"
    quantities_pl |= quantities @ pynini.cdrewrite(pynini.cross("", "er"), "", "[EOS]", NEMO_SIGMA)

    if include_abbr or not itn:
        quantity = quantities | quantities_abbr
        quantities_pl |= quantities_abbr + pynutil.insert("er")
    else:
        quantity = quantities

    one_en = pynini.cross("1", "en")
    one_ett = pynini.cross("1", "ett")
    if itn:
        # accept both here, even if wrong
        one_en = pynini.cross("en", "1")
        one_en |= pynini.cross("ett", "1")

    res = (
        pynutil.insert("integer_part: \"")
        + cardinal_up_to_thousand
        + pynutil.insert("\"")
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(" quantity: \"")
        + quantities_pl
        + pynutil.insert("\"")
    )
    if not itn:
        res |= (
            pynutil.insert("integer_part: \"")
            + cardinal_up_to_thousand_ett
            + pynutil.insert("\"")
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + pynutil.insert(" quantity: \"")
            + "tusen"
            + pynutil.insert("\"")
        )
        res |= (
            pynutil.insert("integer_part: \"")
            + one_ett
            + pynutil.insert("\"")
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + pynutil.insert(" quantity: \"")
            + "tusen"
            + pynutil.insert("\"")
        )
    res |= (
        pynutil.insert("integer_part: \"")
        + one_en
        + pynutil.insert("\"")
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(" quantity: \"")
        + quantity
        + pynutil.insert("\"")
    )
    res |= (
        decimal
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(" quantity: \"")
        + quantities_pl
        + pynutil.insert("\"")
    )
    if not itn:
        res |= (
            decimal_ett
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + pynutil.insert(" quantity: \"")
            + "tusen"
            + pynutil.insert("\"")
        )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -12,5006 biljon -> decimal { negative: "true" integer_part: "tolv"  fractional_part: "fem noll noll sex" quantity: "biljon" }
        1 biljon -> decimal { integer_part: "en" quantity: "biljon" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        cardinal_graph_en = cardinal.graph_en
        cardinal_graph_hundreds_one_non_zero = cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
        cardinal_graph_hundreds_one_non_zero_en = (
            cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one_en
        )
        self.cardinal_graph_hundreds_one_non_zero_en = cardinal_graph_hundreds_one_non_zero_en
        self.cardinal_graph_hundreds_one_non_zero = cardinal_graph_hundreds_one_non_zero

        self.graph = cardinal.two_or_three_digits_read_frac

        self.graph_itn = pynini.invert(cardinal.two_or_three_digits_read_frac_both).optimize()

        if not deterministic:
            self.graph |= cardinal.single_digits_graph.optimize()
            self.graph |= cardinal_graph

        point = pynutil.delete(",")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.graph_integer_en = pynutil.insert("integer_part: \"") + cardinal_graph_en + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(self.graph_integer + pynutil.insert(" "), 0, 1)
            + point
            + pynutil.insert(" ")
            + self.graph_fractional
        )
        self.final_graph_wo_sign = final_graph_wo_sign
        final_graph_wo_sign_en = (
            pynini.closure(self.graph_integer_en + pynutil.insert(" "), 0, 1)
            + point
            + pynutil.insert(" ")
            + self.graph_fractional
        )
        self.final_graph_wo_sign_en = final_graph_wo_sign_en

        quantity_w_abbr = get_quantity(
            final_graph_wo_sign_en,
            final_graph_wo_sign,
            cardinal_graph_hundreds_one_non_zero_en,
            cardinal_graph_hundreds_one_non_zero,
            include_abbr=True,
        )
        quantity_wo_abbr = get_quantity(
            final_graph_wo_sign_en,
            final_graph_wo_sign,
            cardinal_graph_hundreds_one_non_zero_en,
            cardinal_graph_hundreds_one_non_zero,
            include_abbr=False,
        )
        self.final_graph_wo_negative_w_abbr = final_graph_wo_sign | quantity_w_abbr
        self.final_graph_wo_negative_w_abbr_en = final_graph_wo_sign_en | quantity_w_abbr
        self.final_graph_wo_negative = final_graph_wo_sign | quantity_wo_abbr
        self.final_graph_wo_negative_en = final_graph_wo_sign_en | quantity_wo_abbr

        final_graph = optional_graph_negative + self.final_graph_wo_negative_w_abbr

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
