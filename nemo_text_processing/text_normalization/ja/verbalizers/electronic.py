# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NARROW_NON_BREAK_SPACE,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class ElectronicFst(GraphFst):
    """Verbalizes structured Japanese electronic tokens."""

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        symbol = pynini.string_file(get_abs_path("data/electronic/symbol.tsv"))

        def spaced_symbol(written: str):
            return insert_space + (pynini.accep(written) @ symbol) + insert_space

        def insert_spoken_symbol(written: str):
            return insert_space + (pynutil.insert(written) @ symbol) + insert_space

        dot = spaced_symbol(".")
        hyphen = spaced_symbol("-")
        slash = spaced_symbol("/")
        insert_at = insert_spoken_symbol("@")
        insert_colon = insert_spoken_symbol(":")
        insert_slash = insert_spoken_symbol("/")

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        digit_zero_maru = digit | pynini.string_file(get_abs_path("data/numbers/zero_maru.tsv"))
        digit_zero_user = digit | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        special_digit_run = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/electronic/special_digit_runs.tsv")),
            -0.1,
        )

        raw_label = pynini.closure(NEMO_ALPHA | NEMO_DIGIT, 1)
        alpha_label = pynini.closure(NEMO_ALPHA, 1)
        digit_label = pynini.closure(NEMO_DIGIT, 1)

        insert_alpha_digit_space = pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA, NEMO_DIGIT, NEMO_SIGMA)
        insert_digit_alpha_space = pynini.cdrewrite(pynutil.insert(" "), NEMO_DIGIT, NEMO_ALPHA, NEMO_SIGMA)
        alnum_spacing = insert_alpha_digit_space @ insert_digit_alpha_space
        username_reader = pynini.closure(
            NEMO_ALPHA | special_digit_run | digit_zero_user | pynini.accep(" "),
            1,
        )
        raw_mixed_alnum = (
            pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_ALPHA
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_DIGIT
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
        ) | (
            pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_DIGIT
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_ALPHA
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
        )
        username_alnum = (raw_mixed_alnum @ alnum_spacing @ username_reader).optimize()
        username_segment = pynutil.add_weight(username_alnum, -0.1) | alpha_label | digit_label
        username_value = username_segment + pynini.closure((dot | hyphen) + username_segment)

        domain_value = raw_label + pynini.closure(dot + raw_label) + dot + alpha_label
        path_label = raw_label + pynini.closure(hyphen + raw_label)
        path_value = slash + path_label + pynini.closure(slash + path_label)

        username_field = (
            pynutil.delete("username:") + delete_space + pynutil.delete('"') + username_value + pynutil.delete('"')
        )
        domain_field = (
            pynutil.delete("domain:") + delete_space + pynutil.delete('"') + domain_value + pynutil.delete('"')
        )
        protocol_value = pynini.string_file(get_abs_path("data/electronic/protocol.tsv"))
        protocol_field = (
            pynutil.delete("protocol:")
            + delete_space
            + pynutil.delete('"')
            + protocol_value
            + pynutil.delete('"')
            + insert_colon
            + insert_slash
            + insert_slash
        )
        path_field = pynutil.delete("path:") + delete_space + pynutil.delete('"') + path_value + pynutil.delete('"')

        email = username_field + delete_space + insert_at + domain_field
        domain = domain_field
        url = protocol_field + delete_space + domain_field + pynini.closure(delete_space + path_field, 0, 1)

        card_digit = pynini.closure(digit_zero_maru, 1)
        protected_space = pynutil.insert(NEMO_NARROW_NON_BREAK_SPACE)
        card_number_value = card_digit + pynini.closure(pynutil.delete(" ") + protected_space + card_digit)
        card_number_field = (
            pynutil.delete("domain:") + delete_space + pynutil.delete('"') + card_number_value + pynutil.delete('"')
        )
        card_cue = pynini.string_file(get_abs_path("data/electronic/card_cues.tsv")) | (
            pynini.string_file(get_abs_path("data/electronic/card_digit_count_prefix.tsv"))
            + pynini.project(digit, "output")
            + pynini.string_file(get_abs_path("data/electronic/card_digit_count_suffix.tsv"))
        )
        card_cue_field = (
            pynutil.delete("protocol:") + delete_space + pynutil.delete('"') + card_cue + pynutil.delete('"')
        )
        card = pynini.closure(card_cue_field + delete_space, 0, 1) + card_number_field

        filename_stem = pynini.closure(
            pynini.difference(NEMO_NOT_QUOTE, pynini.union(".", "/", "@")),
            1,
        )
        filename = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + filename_stem
            + protected_space
            + pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv"))
            + pynutil.delete('"')
        )

        graph = (email | url | domain | card | filename) + delete_preserve_order
        self.fst = self.delete_tokens(graph.optimize()).optimize()
