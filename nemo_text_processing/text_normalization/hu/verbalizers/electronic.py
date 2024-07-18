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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    at,
    colon,
    delete_preserve_order,
    domain_string,
    double_quotes,
    double_slash,
    http,
    https,
    period,
    protocol_string,
    username_string,
    www,
)
from nemo_text_processing.text_normalization.hu.utils import get_abs_path

digit_no_zero = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv")))
zero = pynini.invert(pynini.string_file(get_abs_path("data/number/zero.tsv")))

graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv"))
server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

accept_space = pynini.accep(NEMO_SPACE)
delete_username = pynutil.delete(username_string + colon + NEMO_SPACE + double_quotes)
delete_double_quotes = pynutil.delete(double_quotes)
delete_domain = pynutil.delete(domain_string + colon + NEMO_SPACE + double_quotes)
delete_protocol = pynutil.delete(protocol_string + colon + NEMO_SPACE + double_quotes)


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. electronic { username: "abc" domain: "hotmail.com" } -> "a b c kukac hotmail pont com"
                                                           -> "a b c kukac h o t m a i l pont c o m"
                                                           -> "a b c kukac hotmail pont c o m"
                                                           -> "a b c at h o t m a i l pont com"
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        graph_digit = digit_no_zero | zero

        def add_space_after_char():
            return pynini.closure(NEMO_NOT_QUOTE - accept_space + pynutil.insert(NEMO_SPACE)) + (
                NEMO_NOT_QUOTE - accept_space
            )

        hungarian_at = [
            "kukacjel ",
            "csiga ",
            "ormány ",
            "farkas á ",
            "bejgli ",
            "at-jel ",
        ]
        at_sign = pynutil.insert("kukac ")
        if not deterministic:
            for sign in hungarian_at:
                at_sign |= pynutil.insert(sign)

        verbalize_characters = pynini.cdrewrite(graph_symbols | graph_digit, "", "", NEMO_SIGMA)

        user_name = delete_username + add_space_after_char() + delete_double_quotes
        user_name @= verbalize_characters

        convert_defaults = pynutil.add_weight(NEMO_NOT_QUOTE, weight=0.0001) | domain_common | server_common
        domain = convert_defaults + pynini.closure(pynutil.insert(NEMO_SPACE) + convert_defaults)
        domain @= verbalize_characters

        domain = delete_domain + domain + delete_double_quotes
        protocol = (
            delete_protocol
            + add_space_after_char() @ pynini.cdrewrite(graph_symbols, "", "", NEMO_SIGMA)
            + delete_double_quotes
        )

        self.graph = (pynini.closure(protocol + NEMO_SPACE, 0, 1) + domain) | (
            user_name + NEMO_SPACE + at_sign + domain | (at_sign + user_name)
        )
        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
