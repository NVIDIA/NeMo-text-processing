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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path


def get_quantity(decimal, cardinal):
    suffix = pynini.union(
        "万",
        "十万",
        "百万",
        "千万",
        "亿",
        "十亿",
        "百亿",
        "千亿",
        "萬",
        "十萬",
        "百萬",
        "千萬",
        "億",
        "十億",
        "百億",
        "千億",
        "拾萬",
        "佰萬",
        "仟萬",
        "拾億",
        "佰億",
        "仟億",
        "拾万",
        "佰万",
        "仟万",
        "仟亿",
        "佰亿",
        "仟亿",
    )
    numbers = cardinal
    res = (
        pynutil.insert('integer_part: "')
        + numbers
        + pynutil.insert('"')
        + pynutil.insert(' quantity: "')
        + suffix
        + pynutil.insert('"')
    )
    res = res | decimal + pynutil.insert(' quantity: "') + suffix + pynutil.insert('"')

    return res


class DecimalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_after_decimal = pynini.string_file(get_abs_path("data/numbers/digit-nano.tsv")) | pynini.closure(
            pynini.cross("零", "0")
        )
        cardinal_before_decimal = cardinal.just_cardinals | pynini.cross("零", "0")

        delete_decimal = pynutil.delete("点") | pynutil.delete("點")

        graph_integer = pynutil.insert('integer_part: "') + cardinal_before_decimal + pynutil.insert('" ')

        graph_string_of_cardinals = pynini.closure(cardinal_after_decimal, 1)
        graph_fractional = pynutil.insert('fractional_part: "') + graph_string_of_cardinals + pynutil.insert('"')

        graph_decimal_no_sign = pynini.closure((graph_integer + delete_decimal + graph_fractional), 1)

        self.final_graph_wo_negative = graph_decimal_no_sign | get_quantity(
            graph_decimal_no_sign, cardinal.just_cardinals
        )

        graph_negative = pynini.cross("负", 'negative: "-" ') | pynini.cross("負", 'negative: "-" ')
        graph_negative = pynini.closure(graph_negative, 0, 1)  # captures only one "负"

        graph_decimal = graph_negative + graph_decimal_no_sign
        graph_decimal = graph_decimal | (graph_negative + get_quantity(graph_decimal_no_sign, cardinal_before_decimal))
        self.final_graph_wo_negative = graph_decimal

        final_graph = self.add_tokens(graph_decimal)
        self.fst = final_graph.optimize()
