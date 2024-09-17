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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path


class CardinalFst(GraphFst):
    def __init__(self):
        """
        Fitite state transducer for classifying cardinals (e.g., 负五十 -> cardinal { negative: "-" integer: "50" })
        This class converts cardinals up to hundred millions (i.e., (10**10))
        Single unit digits are not converted (e.g., 五 -> 五)
        Numbers less than 20 are not converted.
        二十 (2 characters/logograms) is kept as it is but 二十一 (3 characters/logograms) would become 21
        """
        super().__init__(name="cardinal", kind="classify")

        # number of digits to be processed
        delete_hundreds = pynutil.delete("百") | pynutil.delete("佰")
        delete_thousands = pynutil.delete("千") | pynutil.delete("仟")
        closure_ten_thousands = pynini.accep("萬") | pynini.accep("万")
        delete_ten_thousands = pynutil.delete("萬") | pynutil.delete("万")
        closure_hundred_millions = pynini.accep("亿") | pynini.accep("億")
        delete_hundred_millions = pynutil.delete("亿") | pynutil.delete("億")

        # data imported
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        digits = pynini.string_file(get_abs_path("data/numbers/digit-nano.tsv"))
        ties = pynini.string_file(get_abs_path("data/numbers/ties-nano.tsv"))

        # grammar for digits
        graph_digits = digits | pynutil.insert("0")

        # grammar for teens
        ten = pynini.string_map([("十", "1"), ("拾", "1"), ("壹拾", "1"), ("一十", "1")])
        graph_teens = ten + graph_digits

        # grammar for tens, not the output for Cardinal grammar but for pure Arabic digits (used in other grammars)
        graph_tens = (ties + graph_digits) | (pynini.cross(pynini.accep("零"), "0") + graph_digits)
        graph_all = graph_tens | graph_teens | pynutil.insert("00")

        # grammar for hundreds 百
        graph_hundreds_complex = (
            (graph_digits + delete_hundreds + graph_all)
            | (graph_digits + delete_hundreds + pynini.cross(pynini.closure("零"), "0") + graph_digits)
            | (graph_digits + delete_hundreds + graph_teens)
        )
        graph_hundreds = graph_hundreds_complex
        graph_hundreds = graph_hundreds | pynutil.insert("000")

        # grammar for thousands 千
        graph_thousands_complex = (
            (graph_digits + delete_thousands + graph_hundreds_complex)
            | (graph_digits + delete_thousands + pynini.cross(pynini.closure("零"), "0") + graph_all)
            | (graph_digits + delete_thousands + pynini.cross(pynini.closure("零"), "00") + graph_digits)
        )
        graph_thousands = graph_thousands_complex | pynutil.insert("000")

        # grammar for ten thousands 万
        graph_ten_thousands_simple = graph_digits + closure_ten_thousands
        graph_ten_thousands_complex = (
            (graph_digits + delete_ten_thousands + graph_thousands_complex)
            | (graph_digits + delete_ten_thousands + pynini.cross(pynini.closure("零"), "0") + graph_hundreds_complex)
            | (graph_digits + delete_ten_thousands + pynini.cross(pynini.closure("零"), "00") + graph_all)
            | (graph_digits + delete_ten_thousands + pynini.cross(pynini.closure("零"), "000") + graph_digits)
        )
        graph_ten_thousands = (
            pynutil.add_weight(graph_ten_thousands_simple, -1.0)
            | graph_ten_thousands_complex
            | pynutil.insert("00000")
        )

        # grammmar for hundred thousands 十万
        graph_hundred_thousands_simple = graph_all + closure_ten_thousands
        graph_hundred_thousands_complex = (
            (graph_all + delete_ten_thousands + graph_thousands_complex)
            | (graph_all + delete_ten_thousands + pynini.cross(pynini.closure("零"), "0") + graph_hundreds_complex)
            | (graph_all + delete_ten_thousands + pynini.cross(pynini.closure("零"), "00") + graph_all)
            | (graph_all + delete_ten_thousands + pynini.cross(pynini.closure("零"), "000") + graph_digits)
        )
        graph_hundred_thousands = (
            pynutil.add_weight(graph_hundred_thousands_simple, -1.0)
            | graph_hundred_thousands_complex
            | pynutil.insert("000000")
        )

        # grammar for millions 百万
        graph_millions_simple = graph_hundreds_complex + closure_ten_thousands
        graph_millions_complex = (
            (graph_hundreds_complex + delete_ten_thousands + graph_thousands_complex)
            | (
                graph_hundreds_complex
                + delete_ten_thousands
                + pynini.cross(pynini.closure("零"), "0")
                + graph_hundreds_complex
            )
            | (graph_hundreds_complex + delete_ten_thousands + pynini.cross(pynini.closure("零"), "00") + graph_all)
            | (graph_hundreds_complex + delete_ten_thousands + pynini.cross(pynini.closure("零"), "000") + graph_digits)
        )
        graph_millions = (
            pynutil.add_weight(graph_millions_simple, -1.0) | graph_millions_complex | pynutil.insert("0000000")
        )

        # grammar for ten millions 千万
        graph_ten_millions_simple = graph_thousands_complex + closure_ten_thousands
        graph_ten_millions_complex = (
            (graph_thousands_complex + delete_ten_thousands + graph_thousands_complex)
            | (
                graph_thousands_complex
                + delete_ten_thousands
                + pynini.cross(pynini.closure("零"), "0")
                + graph_hundreds_complex
            )
            | (graph_thousands_complex + delete_ten_thousands + pynini.cross(pynini.closure("零"), "00") + graph_all)
            | (
                graph_thousands_complex
                + delete_ten_thousands
                + pynini.cross(pynini.closure("零"), "000")
                + graph_digits
            )
        )
        graph_ten_millions = pynutil.add_weight(graph_ten_millions_simple, -1.0) | graph_ten_millions_complex
        graph_ten_millions = graph_ten_millions | pynutil.insert("00000000")

        # grammar for hundred millions 亿
        graph_hundred_millions_simple = graph_digits + closure_hundred_millions
        graph_hundred_millions_complex = (
            (graph_digits + delete_hundred_millions + graph_ten_millions_complex)
            | (
                graph_digits
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0")
                + graph_millions_complex
            )
            | (
                graph_digits
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00")
                + graph_hundred_thousands_complex
            )
            | (
                graph_digits
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "000")
                + graph_ten_thousands_complex
            )
            | (
                graph_digits
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0000")
                + graph_thousands_complex
            )
            | (
                graph_digits
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00000")
                + graph_hundreds_complex
            )
            | (graph_digits + delete_hundred_millions + pynini.cross(pynini.closure("零"), "000000") + graph_all)
            | (graph_digits + delete_hundred_millions + pynini.cross(pynini.closure("零"), "0000000") + graph_digits)
        )
        graph_hundred_millions = (
            pynutil.add_weight(graph_hundred_millions_simple, -1.0)
            | graph_hundred_millions_complex
            | pynutil.insert("000000000")
        )

        # grammar for billions 十亿
        graph_billions_simple = graph_all + closure_hundred_millions
        graph_billions_complex = (
            (graph_all + delete_hundred_millions + graph_ten_millions_complex)
            | (graph_all + delete_hundred_millions + pynini.cross(pynini.closure("零"), "0") + graph_millions_complex)
            | (
                graph_all
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00")
                + graph_hundred_thousands_complex
            )
            | (
                graph_all
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "000")
                + graph_ten_thousands_complex
            )
            | (
                graph_all
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0000")
                + graph_thousands_complex
            )
            | (
                graph_all
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00000")
                + graph_hundreds_complex
            )
            | (graph_all + delete_hundred_millions + pynini.cross(pynini.closure("零"), "000000") + graph_all)
            | (graph_all + delete_hundred_millions + pynini.cross(pynini.closure("零"), "0000000") + graph_digits)
        )
        graph_billions = (
            pynutil.add_weight(graph_billions_simple, -1.0) | graph_billions_complex | pynutil.insert("0000000000")
        )

        # grammar for ten billions 百亿
        graph_ten_billions_simple = graph_hundreds_complex + closure_hundred_millions
        graph_ten_billions_complex = (
            (graph_hundreds_complex + delete_hundred_millions + graph_ten_millions_complex)
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0")
                + graph_millions_complex
            )
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00")
                + graph_hundred_thousands_complex
            )
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "000")
                + graph_ten_thousands_complex
            )
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0000")
                + graph_thousands_complex
            )
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00000")
                + graph_hundreds_complex
            )
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "000000")
                + graph_all
            )
            | (
                graph_hundreds_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0000000")
                + graph_digits
            )
        )
        graph_ten_billions = (
            pynutil.add_weight(graph_ten_billions_simple, -1.0)
            | graph_ten_billions_complex
            | pynutil.insert("00000000000")
        )

        # grammar for hundred billions 千亿
        graph_hundred_billions_simple = graph_thousands_complex + closure_hundred_millions
        graph_hundred_billions_complex = (
            (graph_thousands_complex + delete_hundred_millions + graph_ten_millions_complex)
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0")
                + graph_millions_complex
            )
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00")
                + graph_hundred_thousands_complex
            )
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "000")
                + graph_ten_thousands_complex
            )
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0000")
                + graph_thousands_complex
            )
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "00000")
                + graph_hundreds_complex
            )
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "000000")
                + graph_all
            )
            | (
                graph_thousands_complex
                + delete_hundred_millions
                + pynini.cross(pynini.closure("零"), "0000000")
                + graph_digits
            )
        )
        graph_hundred_billions = (
            pynutil.add_weight(graph_hundred_billions_simple, -1.0) | graph_hundred_billions_complex
        )

        # combining grammar; output for cardinal grammar
        graph = pynini.union(
            graph_hundred_billions,
            graph_ten_billions,
            graph_billions,
            graph_hundred_millions,
            graph_ten_millions,
            graph_millions,
            graph_hundred_thousands,
            graph_ten_thousands,
            graph_thousands,
            graph_hundreds,
            graph_all,
            graph_teens,
            graph_digits,
            zero,
        )

        # combining grammar; output consists only arabic numbers
        graph_just_cardinals = pynini.union(
            graph_hundred_billions_complex,
            graph_ten_billions_complex,
            graph_billions_complex,
            graph_hundred_millions_complex,
            graph_ten_millions_complex,
            graph_millions_complex,
            graph_hundred_thousands_complex,
            graph_ten_thousands_complex,
            graph_thousands_complex,
            graph_hundreds_complex,
            graph_all,
            graph_teens,
            graph_digits,
            zero,
        )

        # delete unnecessary leading zero
        delete_leading_zeros = pynutil.delete(pynini.closure("0"))
        stop_at_non_zero = pynini.difference(NEMO_DIGIT, "0")
        rest_of_cardinal = pynini.closure(NEMO_DIGIT) | pynini.closure(NEMO_SIGMA)

        # output for cardinal grammar without leading zero
        clean_cardinal = delete_leading_zeros + stop_at_non_zero + rest_of_cardinal
        clean_cardinal = clean_cardinal | "0"
        graph = graph @ clean_cardinal  # output for regular cardinals
        self.for_ordinals = graph  # used for ordinal grammars

        # output for pure arabic number without leading zero
        clean_just_cardinal = delete_leading_zeros + stop_at_non_zero + rest_of_cardinal
        clean_just_cardinal = clean_just_cardinal | "0"
        graph_just_cardinals = graph_just_cardinals @ clean_just_cardinal  # output for other grammars
        self.just_cardinals = graph_just_cardinals  # used for other grammars

        # final grammar for cardinal output; tokenization
        optional_minus_graph = (pynini.closure(pynutil.insert("negative: ") + pynini.cross("负", '"-"'))) | (
            pynini.closure(pynutil.insert("negative: ") + pynini.cross("負", '"-"'))
        )
        final_graph = optional_minus_graph + pynutil.insert('integer: "') + graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
