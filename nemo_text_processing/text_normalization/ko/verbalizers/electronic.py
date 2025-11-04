# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer (FST) for verbalizing **electronic expressions** (email/URL/domain).

    Input tokens:
        tokens { electronic { username: "abc" domain: "abc.com" } }

    Example output (policy-dependent):
        abc 골뱅이 abc 닷컴

    Args:
        deterministic: If True, produce a single verbalization.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        # 1) Handle digits (0–9)
        graph_digit_no_zero = pynini.string_file(get_abs_path("data/number/digit.tsv")).optimize()

        graph_zero = pynini.cross("0", "영")
        if not deterministic:
            graph_zero |= pynini.cross("0", "공")
        graph_digit = (graph_digit_no_zero | graph_zero).optimize()

        digit_inline_rewrite = pynini.cdrewrite(
            graph_digit,
            "",
            "",
            NEMO_SIGMA,
        )

        # 2) Load electronic symbols (ex: "." → "점")
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        NEMO_NOT_BRACKET = pynini.difference(NEMO_CHAR, pynini.union("{", "}")).optimize()

        # 3) Default spacing for characters, symbols, and digits
        default_chars_symbols = pynini.cdrewrite(
            pynutil.insert(" ") + (graph_symbols | graph_digit | NEMO_ALPHA) + pynutil.insert(" "),
            "",
            "",
            NEMO_SIGMA,
        )
        default_chars_map = pynini.compose(pynini.closure(NEMO_NOT_BRACKET), default_chars_symbols).optimize()

        # 4) username part (add spaces between characters)
        raw_username = pynini.closure(NEMO_NOT_QUOTE, 1)

        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete('"')
            + (raw_username @ digit_inline_rewrite)
            + pynutil.delete('"')
        )

        # 5) domain part (handle common endings like .com → 닷컴)
        domain_common_pairs = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).optimize()

        # Rewrite known domains (.com → 닷컴)
        tld_rewrite = pynini.cdrewrite(
            domain_common_pairs,
            "",
            "",
            NEMO_SIGMA,
        )
        # Add a space before “닷” if needed
        add_space_before_dot = pynini.cdrewrite(
            pynini.cross("닷", " 닷"),
            (NEMO_ALPHA | NEMO_DIGIT | NEMO_CHAR),
            "",
            NEMO_SIGMA,
        )

        raw_domain = pynini.closure(NEMO_NOT_QUOTE, 1)

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + ((raw_domain @ digit_inline_rewrite) @ tld_rewrite @ add_space_before_dot)
            + delete_space
            + pynutil.delete('"')
        ).optimize()

        # 6) protocol (like “https://” or “file:///”)
        protocol = (
            pynutil.delete('protocol: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"') + insert_space
        )

        # 7) Combine: optional protocol + optional username + domain
        graph = (
            pynini.closure(protocol + delete_space, 0, 1)
            + pynini.closure(user_name + delete_space + pynutil.insert(" 골뱅이 ") + delete_space, 0, 1)
            + domain
            + delete_space
        ).optimize() @ pynini.cdrewrite(delete_extra_space, "", "", NEMO_SIGMA)
        self.fst = self.delete_tokens(graph).optimize()
