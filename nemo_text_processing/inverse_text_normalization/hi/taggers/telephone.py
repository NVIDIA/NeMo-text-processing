# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    DIGIT_GLYPH_TO_ASCII,
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    load_symbols,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path

shunya = (
    pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv")).invert()
)
digit_without_shunya = (
    pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv")).invert()
)
digit = digit_without_shunya | shunya


def get_context(keywords: list):
    keywords = pynini.union(*keywords)

    all_digits = pynini.project(digit, "input")

    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + NEMO_WHITE_SPACE
    window = pynini.closure(word, 0, 5)
    before = (keywords + window).optimize()
    after = (window + keywords).optimize()

    return before, after


def generate_context_graph(context_keywords, length):
    context_before, context_after = get_context(context_keywords)
    digits = pynini.closure(digit + delete_space, length - 1, length - 1) + digit

    graph_after_context = digits + NEMO_WHITE_SPACE + context_after
    graph_before_context = context_before + NEMO_WHITE_SPACE + digits
    graph_without_context = digits

    return (
        pynutil.insert("number_part: \"")
        + (graph_before_context | graph_after_context | graph_without_context)
        + pynutil.insert("\" ")
    ).optimize()


def generate_pincode(context_keywords):
    return generate_context_graph(context_keywords, 6)


def generate_credit(context_keywords):
    return generate_context_graph(context_keywords, 4)


def generate_mobile(context_keywords):
    context_before, context_after = get_context(context_keywords)

    country_code = pynini.cross("प्लस", "+") + pynini.closure(delete_space + digit, 2, 2) + NEMO_WHITE_SPACE
    graph_country_code = (
        pynutil.insert("country_code: \"")
        + (context_before + NEMO_WHITE_SPACE) ** (0, 1)
        + country_code
        + pynutil.insert("\" ")
    )

    number_part = digit_without_shunya + delete_space + pynini.closure(digit + delete_space, 8, 8) + digit
    graph_number = (
        pynutil.insert("number_part: \"")
        + number_part
        + pynini.closure(NEMO_WHITE_SPACE + context_after, 0, 1)
        + pynutil.insert("\" ")
    )

    graph = (graph_country_code + graph_number) | graph_number
    return graph.optimize()


def generate_telephone(context_keywords):
    context_before, context_after = get_context(context_keywords)

    landline = shunya + delete_space + pynini.closure(digit + delete_space, 9, 9) + digit
    landline_with_context_before = context_before + NEMO_WHITE_SPACE + landline
    landline_with_context_after = landline + NEMO_WHITE_SPACE + context_after

    return (
        pynutil.insert("number_part: \"")
        + (landline | landline_with_context_before | landline_with_context_after)
        + pynutil.insert("\" ")
    )


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
    e.g. प्लस इक्यानवे नौ आठ सात छह पांच चार तीन दो एक शून्य => tokens { name: "+९१ ९८७६५ ४३२१०" }
    This class also supports IP addresses, e.g.
    e.g. एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक => tokens { telephone { number_part: "192.168.1.1" } }
    Args:
        Cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        # Load context cues from TSV file
        context_cues = pynini.string_file(get_abs_path("data/telephone/context_cues.tsv"))

        def keywords(category):
            return pynini.compose(pynutil.delete(category), context_cues).project("output").optimize()

        mobile = generate_mobile([keywords("mobile")])
        landline = generate_telephone([keywords("landline")])
        pincode = generate_pincode([keywords("pincode")])
        credit = generate_credit([keywords("credit")])

        sym = load_symbols(get_abs_path("data/electronic/symbols.tsv"))
        ip_dot = delete_space + sym["dot"] + delete_space
        ip_digit = pynini.compose(digit, DIGIT_GLYPH_TO_ASCII) | DIGIT_GLYPH_TO_ASCII
        ip_octet = ip_digit + pynini.closure(delete_space + ip_digit, 0, 2)
        ip_graph = pynutil.insert("number_part: \"") + ip_octet + (ip_dot + ip_octet) ** 3 + pynutil.insert("\" ")

        graph = (
            pynutil.add_weight(mobile, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit, 0.9)
            | pynutil.add_weight(pincode, 1)
            | pynutil.add_weight(ip_graph, 0.7)
        )

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
