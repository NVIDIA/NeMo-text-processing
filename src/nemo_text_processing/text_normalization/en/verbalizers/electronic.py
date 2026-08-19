# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_LOWER,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> c d f one at a b c dot e d u

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit_no_zero = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv"))).optimize()
        graph_zero = pynini.cross("0", "zero")
        long_numbers = pynutil.add_weight(graph_digit_no_zero + pynini.cross("000", " thousand"), MIN_NEG_WEIGHT)

        if not deterministic:
            graph_zero |= pynini.cross("0", "o") | pynini.cross("0", "oh")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        NEMO_NOT_BRACKET = pynini.difference(NEMO_CHAR, pynini.union("{", "}")).optimize()
        dict_words = pynini.project(pynini.string_file(get_abs_path("data/electronic/words.tsv")), "output")
        default_chars_symbols = pynini.cdrewrite(
            pynutil.insert(" ") + (graph_symbols | graph_digit | long_numbers) + pynutil.insert(" "),
            "",
            "",
            NEMO_SIGMA,
        )
        default_chars_symbols = pynini.compose(
            pynini.closure(NEMO_NOT_BRACKET), default_chars_symbols.optimize()
        ).optimize()

        # this is far cases when user name was split by dictionary words, i.e. "sevicepart@ab.com" -> "service part"
        space_separated_dict_words = pynutil.add_weight(
            NEMO_ALPHA
            + pynini.closure(NEMO_ALPHA | NEMO_SPACE)
            + NEMO_SPACE
            + pynini.closure(NEMO_ALPHA | NEMO_SPACE),
            MIN_NEG_WEIGHT,
        )

        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + (default_chars_symbols | space_separated_dict_words).optimize()
            + pynutil.delete("\"")
        )

        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        # this will be used for a safe fallback
        domain_all = pynini.compose(
            default_chars_symbols,
            pynini.closure(TO_LOWER | NEMO_LOWER | NEMO_SPACE | pynutil.add_weight(dict_words, MIN_NEG_WEIGHT)),
        )

        domain = (
            domain_all
            + insert_space
            + plurals._priority_union(
                domain_common, pynutil.add_weight(pynini.cross(".", "dot"), weight=0.0001), NEMO_SIGMA
            )
            + pynini.closure(insert_space + default_chars_symbols, 0, 1)
        )

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + (domain | pynutil.add_weight(domain_all, weight=100)).optimize()
            + delete_space
            + pynutil.delete("\"")
        ).optimize()

        protocol = pynutil.delete("protocol: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph = (
            pynini.closure(protocol + delete_space, 0, 1)
            + pynini.closure(user_name + delete_space + pynutil.insert(" at ") + delete_space, 0, 1)
            + domain
            + delete_space
        ).optimize() @ pynini.cdrewrite(delete_extra_space, "", "", NEMO_SIGMA)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
