# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers and IP addresses.

    Supported formats:

    1. Basic telephone (Vietnamese mobile with formatting):
        "không chín ba sáu năm năm năm bốn bốn chín"
        -> telephone { number_part: "093-655-5449" }

    2. International format with country code:
        "cộng tám mươi bốn không chín ba sáu năm năm năm bốn bốn chín"
        -> telephone { country_code: "+84" number_part: "093-655-5449" }

    3. IP addresses (using "chấm" for dot):
        "một chín hai chấm một sáu tám chấm không chấm một"
        -> telephone { number_part: "192.168.0.1" }

    4. Emergency/hotline numbers:
        "một một hai" -> telephone { number_part: "112" }

    5. Credit card (15-16 digits):
        "một hai ba bốn năm sáu bảy tám chín mười một hai ba bốn năm sáu"
        -> telephone { number_part: "1234 5678 9101 2345" }

    Args:
        cardinal: CardinalFst - required for parsing multi-digit numbers like "tám mươi bốn"
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_digit_special = pynini.string_file(get_abs_path("data/numbers/digit_special.tsv"))
        digit = pynini.union(graph_digit, graph_zero)
        last_digit = pynini.union(digit, graph_digit_special)
        cardinal_two_digit = pynini.compose(cardinal.graph_no_exception, NEMO_DIGIT**2)

        vietnamese_mobile = pynini.compose(
            pynini.cross("không", "0") + delete_space + pynini.closure(digit + delete_space, 8) + last_digit,
            pynini.accep("0")
            + NEMO_DIGIT**2
            + pynutil.insert("-")
            + NEMO_DIGIT**3
            + pynutil.insert("-")
            + NEMO_DIGIT**4,
        )

        basic_digits = pynini.closure(digit + delete_space, 2) + last_digit

        basic_phone = pynini.union(pynutil.add_weight(vietnamese_mobile, -0.01), basic_digits)

        country_code_digits = pynini.union(
            pynutil.add_weight(cardinal_two_digit, -0.001), digit + delete_space + digit, digit
        )

        phone_with_country_code = (
            pynutil.insert('country_code: "')
            + pynini.cross("cộng ", "+")
            + country_code_digits
            + pynutil.insert('"')
            + delete_space
            + insert_space
            + pynutil.insert('number_part: "')
            + basic_phone
            + pynutil.insert('"')
        )

        phone_basic = pynutil.insert('number_part: "') + basic_phone + pynutil.insert('"')

        basic_phone_graph = pynini.union(pynutil.add_weight(phone_with_country_code, -0.1), phone_basic)

        ip_octet = pynini.union(pynini.closure(digit + delete_space, 0, 2) + digit, cardinal_two_digit)

        ip_graph = ip_octet + (delete_space + pynini.cross("chấm", ".") + delete_space + ip_octet) ** 3

        ip_with_tag = pynutil.insert('number_part: "') + ip_graph + pynutil.insert('"')

        sixteen_digits = pynini.closure(digit + delete_space, 15) + digit
        card_16 = pynini.compose(
            sixteen_digits,
            NEMO_DIGIT**4 + insert_space + NEMO_DIGIT**4 + insert_space + NEMO_DIGIT**4 + insert_space + NEMO_DIGIT**4,
        )

        fifteen_digits = pynini.closure(digit + delete_space, 14) + digit
        card_15 = pynini.compose(
            fifteen_digits, NEMO_DIGIT**4 + insert_space + NEMO_DIGIT**6 + insert_space + NEMO_DIGIT**5
        )

        card_graph = pynini.union(card_16, card_15)
        card_with_tag = pynutil.insert('number_part: "') + card_graph + pynutil.insert('"')

        graph = pynini.union(
            pynutil.add_weight(ip_with_tag, weight=-0.01),
            pynutil.add_weight(card_with_tag, weight=-0.005),
            basic_phone_graph,
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
