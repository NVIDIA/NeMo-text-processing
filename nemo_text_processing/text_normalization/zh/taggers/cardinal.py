# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. 23 -> cardinal { integer: "二十三" }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_digit_alt = pynini.string_file(get_abs_path("data/number/digit_alt.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/number/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/number/teen.tsv"))
        graph_teen_alt = pynini.string_file(get_abs_path("data/number/teen_alt.tsv"))

        alls = NEMO_DIGIT ** 2 | NEMO_DIGIT ** 1
        graph_all = (
            (graph_ties + (graph_digit | pynutil.delete('0'))) | graph_teen_alt | graph_digit
        )  # graph_all when within a larger number e.g., 316-> 三百一十六 instead of 三百十六

        graph_all = alls @ graph_all
        graph_all_alt = (
            (graph_ties + (graph_digit | pynutil.delete('0'))) | graph_teen | graph_digit
        )  # graph_all when at the head of the larger numbere.g., 13万 -> 十三万 instead of 一十三万
        graph_all_alt = alls @ graph_all_alt

        hundreds = NEMO_DIGIT ** 3
        graph_hundred_component = (graph_digit + pynutil.insert('百')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            (pynini.closure(pynutil.delete('0') + pynutil.insert('零')) + graph_all),
        )
        graph_hundred = hundreds @ graph_hundred_component

        self.digit = graph_digit.optimize()
        self.all = graph_all.optimize()

        thousands = NEMO_DIGIT ** 4
        graph_thousand_component = (graph_digit_alt + pynutil.insert('千')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_hundred_component,
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_thousand = thousands @ graph_thousand_component

        ten_thousands = NEMO_DIGIT ** 5
        graph_ten_thousand_component = (graph_digit_alt + pynutil.insert('万')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_ten_thousand = ten_thousands @ graph_ten_thousand_component

        hundred_thousands = NEMO_DIGIT ** 6
        hundred_thousands_position = NEMO_DIGIT ** 2
        hundred_thousands_position = hundred_thousands_position @ graph_all_alt
        graph_hundred_thousand_component = (hundred_thousands_position + pynutil.insert('万')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_hundred_thousand = hundred_thousands @ graph_hundred_thousand_component

        millions = NEMO_DIGIT ** 7
        million_position = NEMO_DIGIT ** 3
        million_position = million_position @ graph_hundred_component
        graph_million_component = (million_position + pynutil.insert('万')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_million = millions @ graph_million_component

        ten_millions = NEMO_DIGIT ** 8
        ten_million_position = NEMO_DIGIT ** 4
        ten_million_position = ten_million_position @ graph_thousand_component
        graph_ten_million_component = (ten_million_position + pynutil.insert('万')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_thousand_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_ten_million = ten_millions @ graph_ten_million_component

        hundred_millions = NEMO_DIGIT ** 9
        graph_hundred_million_component = (graph_digit_alt + pynutil.insert('亿')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_million_component),
            (pynutil.delete('00') + pynutil.insert('零') + graph_hundred_thousand_component),
            (pynutil.delete('000') + pynutil.insert('零') + graph_ten_thousand_component),
            (pynutil.delete('0000') + pynutil.insert('零') + graph_thousand_component),
            (pynutil.delete('00000') + pynutil.insert('零') + graph_hundred_component),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_hundred_million = hundred_millions @ graph_hundred_million_component

        thousand_millions = NEMO_DIGIT ** 10
        thousand_millions_position = NEMO_DIGIT ** 2
        thousand_millions_position = thousand_millions_position @ graph_all_alt
        graph_thousand_million_component = (thousand_millions_position + pynutil.insert('亿')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_million_component),
            (pynutil.delete('00') + pynutil.insert('零') + graph_hundred_thousand_component),
            (pynutil.delete('000') + pynutil.insert('零') + graph_ten_thousand_component),
            (pynutil.delete('0000') + pynutil.insert('零') + graph_thousand_component),
            ((pynutil.delete('00000') + pynutil.insert('零') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_thousand_million = thousand_millions @ graph_thousand_million_component

        ten_billions = NEMO_DIGIT ** 11
        ten_billions_position = NEMO_DIGIT ** 3
        ten_billions_position = ten_billions_position @ graph_hundred_component
        graph_ten_billions_component = (ten_billions_position + pynutil.insert('亿')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_million_component),
            (pynutil.delete('00') + pynutil.insert('零') + graph_hundred_thousand_component),
            (pynutil.delete('000') + pynutil.insert('零') + graph_ten_thousand_component),
            (pynutil.delete('0000') + pynutil.insert('零') + graph_thousand_component),
            ((pynutil.delete('00000') + pynutil.insert('零') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_ten_billions = ten_billions @ graph_ten_billions_component

        hundred_billions = NEMO_DIGIT ** 12
        hundred_billions_position = NEMO_DIGIT ** 4
        hundred_billions_position = hundred_billions_position @ graph_thousand_component
        graph_hundred_billions_component = (hundred_billions_position + pynutil.insert('亿')) + pynini.union(
            pynini.closure(pynutil.delete('0')),
            graph_ten_million_component,
            (pynutil.delete('0') + pynutil.insert('零') + graph_million_component),
            (pynutil.delete('00') + pynutil.insert('零') + graph_hundred_thousand_component),
            (pynutil.delete('000') + pynutil.insert('零') + graph_ten_thousand_component),
            (pynutil.delete('0000') + pynutil.insert('零') + graph_thousand_component),
            ((pynutil.delete('00000') + pynutil.insert('零') + graph_hundred_component)),
            (pynini.closure(pynutil.delete('0')) + pynutil.insert('零') + graph_all),
        )
        graph_hundred_billions = hundred_billions @ graph_hundred_billions_component

        graph = pynini.union(
            graph_hundred_billions,
            graph_ten_billions,
            graph_thousand_million,
            graph_hundred_million,
            graph_ten_million,
            graph_million,
            graph_hundred_thousand,
            graph_ten_thousand,
            graph_thousand,
            graph_hundred,
            graph_all_alt,
            graph_zero,
        )
        self.just_cardinals = graph.optimize()
        optional_sign = (
            pynutil.insert("negative: \"") + (pynini.accep("-") | pynini.cross("负", "-")) + pynutil.insert("\"")
        )
        final_graph = (
            optional_sign + pynutil.insert(" ") + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        ) | (pynutil.insert("integer: \"") + graph + pynutil.insert("\""))

        self.with_sign = final_graph.optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
