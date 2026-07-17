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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_ALPHA, NEMO_DIGIT, NEMO_NOT_SPACE, GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying Japanese electronic expressions.

    Examples:
        abc@abc.com -> electronic { username: "abc" domain: "abc.com" preserve_order: true }
        https://www.nvidia.com
        -> electronic { protocol: "https" domain: "www.nvidia.com" preserve_order: true }
        1234-5678-9012-3456
        -> electronic { domain: "1234 5678 9012 3456" preserve_order: true }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        alnum = NEMO_ALPHA | NEMO_DIGIT
        hyphen = pynini.accep("-")
        dot = pynini.accep(".")
        slash = pynini.accep("/")
        at = pynini.accep("@")

        label = pynini.closure(alnum | hyphen, 1)
        tld = pynini.closure(NEMO_ALPHA, 2)
        domain_core = label + pynini.closure(dot + label) + dot + tld
        domain_field = pynutil.insert('domain: "') + domain_core + pynutil.insert('"')

        username_symbol = dot | hyphen
        username_core = alnum + pynini.closure(alnum | username_symbol)
        username_field = (
            pynutil.insert('username: "')
            + username_core
            + pynutil.insert('"')
            + pynutil.delete("@")
            + pynutil.insert(" ")
        )
        email = username_field + domain_field

        protocol = pynini.string_file(get_abs_path("data/electronic/protocol.tsv"))
        protocol_field = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        path_segment = pynini.closure(alnum | hyphen, 1)
        path_core = slash + path_segment + pynini.closure(slash + path_segment)
        path_field = pynutil.insert(' path: "') + path_core + pynutil.insert('"')
        url = (
            protocol_field
            + pynutil.delete("://")
            + pynutil.insert(" ")
            + domain_field
            + pynini.closure(path_field, 0, 1)
        )

        four_digits = NEMO_DIGIT**4
        card_separator = pynutil.delete("-") | pynutil.delete(" ")
        grouped_card_number = (
            four_digits
            + card_separator
            + pynutil.insert(" ")
            + four_digits
            + card_separator
            + pynutil.insert(" ")
            + four_digits
            + card_separator
            + pynutil.insert(" ")
            + four_digits
        )
        card_number_field = pynutil.insert('domain: "') + grouped_card_number + pynutil.insert('"')
        short_card_number_field = pynutil.insert('domain: "') + four_digits + pynutil.insert('"')

        card_cue = pynini.string_file(get_abs_path("data/electronic/card_cues.tsv"))
        card_cue_field = pynutil.insert('protocol: "') + card_cue + pynutil.insert('" ')
        credit_card = card_number_field
        card_with_cue = card_cue_field + card_number_field
        card_tail_with_cue = card_cue_field + short_card_number_field

        digit_count_prefix = pynini.string_file(get_abs_path("data/electronic/card_digit_count_prefix.tsv"))
        digit_count_suffix = pynini.string_file(get_abs_path("data/electronic/card_digit_count_suffix.tsv"))
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        card_tail = (
            pynutil.insert('protocol: "')
            + digit_count_prefix
            + digit
            + digit_count_suffix
            + pynutil.insert('" ')
            + short_card_number_field
        )

        extension = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv")),
            "input",
        )
        filename_stem = pynini.closure(
            pynini.difference(NEMO_NOT_SPACE, pynini.union(dot, slash, at)),
            1,
        )
        filename = pynutil.insert('domain: "') + filename_stem + extension + pynutil.insert('"')

        graph = (
            pynutil.add_weight(credit_card, -0.1)
            | pynutil.add_weight(card_with_cue, -0.1)
            | pynutil.add_weight(card_tail_with_cue, -0.1)
            | card_tail
            | email
            | url
            | domain_field
            | filename
        )
        graph += pynutil.insert(" preserve_order: true")

        self.fst = self.add_tokens(graph.optimize()).optimize()
