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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path

digit_no_zero = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv"))
server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))
spoken_unit = pynini.string_file(get_abs_path("data/electronic/electronic_spoken_unit.tsv"))


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        graph_digit = digit_no_zero | zero

        def add_space_after_char():
            return pynini.closure(NEMO_NOT_QUOTE - pynini.accep(NEMO_SPACE) + pynutil.insert(NEMO_SPACE)) + (
                NEMO_NOT_QUOTE - pynini.accep(NEMO_SPACE)
            )

        verbalize_characters = pynini.cdrewrite(graph_symbols | graph_digit, "", "", NEMO_SIGMA)

        # Prefer whole tokens (server names, TLDs, company/common words) over letter-by-letter.
        user_segment = (
            pynutil.add_weight(NEMO_NOT_QUOTE, weight=0.0001) | server_common | spoken_unit
        )
        user_name = (
            pynutil.delete('username: "')
            + (user_segment + pynini.closure(pynutil.insert(NEMO_SPACE) + user_segment))
            + pynutil.delete('"')
        )
        user_name @= verbalize_characters

        convert_defaults = (
            pynutil.add_weight(NEMO_NOT_QUOTE, weight=0.0001) | domain_common | server_common | spoken_unit
        )
        domain = convert_defaults + pynini.closure(pynutil.insert(NEMO_SPACE) + convert_defaults)
        domain @= verbalize_characters
        domain = pynutil.delete('domain: "') + domain + pynutil.delete('"')

        protocol = (
            pynutil.delete('protocol: "')
            + add_space_after_char() @ pynini.cdrewrite(graph_symbols, "", "", NEMO_SIGMA)
            + pynutil.delete('"')
        )

        self.graph = (pynini.closure(protocol + NEMO_SPACE, 0, 1) + domain) | (
            user_name + NEMO_SPACE + pynutil.insert("arroba" + NEMO_SPACE) + domain
            | (pynutil.insert("arroba" + NEMO_SPACE) + user_name)
        )

        self.fst = self.delete_tokens(self.graph + delete_preserve_order).optimize()
