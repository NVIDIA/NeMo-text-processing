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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer (FST) for classifying **electronic expressions** such as
    email addresses, URLs, and domain names in Korean.

    Example conversions:
        - abc@nvidia.co.kr  →  electronic { username: "abc" domain: "nvidia.co.kr" }
        - www.nvidia.com    →  electronic { domain: "www.nvidia.com" }
        - https://nvidia.com → electronic { protocol: "HTTPS colon slash slash" domain: "nvidia.com" }
        - 1234-5678-9012-3456 → electronic { protocol: "credit card" domain: "1234567890123456" }

    Args:
        cardinal:  FST for digit/number verbalization (used for numeric parts if non-deterministic).
        deterministic:  If True, provides a single transduction path; otherwise allows multiple.
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # ---------- Basic character ranges and symbols ----------
        LOWER = pynini.union(*[pynini.accep(c) for c in "abcdefghijklmnopqrstuvwxyz"])
        UPPER = pynini.union(*[pynini.accep(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
        ASCII_LETTER = (LOWER | UPPER).optimize()
        ASCII_ALNUM = (ASCII_LETTER | NEMO_DIGIT).optimize()

        HYPHEN = pynini.accep("-")
        DOT = pynini.accep(".")
        SLASH = pynini.accep("/")
        AT = pynini.accep("@")

        # Handle numeric reading mode (only for non-deterministic mode)
        numbers = NEMO_DIGIT if deterministic else (pynutil.insert(NEMO_SPACE) + cardinal.long_numbers + pynutil.insert(NEMO_SPACE))

        # ---------- Load resources ----------
        cc_cues = pynini.string_file(get_abs_path("data/electronic/cc_cues.tsv"))
        accepted_symbols = pynini.project(pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input")
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        # ---------- Username ----------
        # Exclude '@' from username
        username_symbols = pynini.difference(accepted_symbols, AT)
        # Start with alphanumeric and allow symbols/numbers repeatedly
        username_core = ASCII_ALNUM + pynini.closure(ASCII_ALNUM | numbers | username_symbols)
        username = pynutil.insert('username: "') + username_core + pynutil.insert('"') + pynini.cross("@", NEMO_SPACE)

        # ---------- Domain ----------
        # Simplified RFC: label = [A-Za-z0-9-]+ , TLD = '.' [A-Za-z0-9]{2,}
        label = pynini.closure(ASCII_ALNUM | HYPHEN, 1)
        tld = DOT + pynini.closure(ASCII_ALNUM, 2)
        # Domain can be (label + TLD) or TLD only (e.g., ".com")
        domain_core = (label + pynini.closure(tld, 1)) | tld

        # Optional path after domain (e.g., /path)
        path_segment = pynini.closure(NEMO_NOT_SPACE, 1)   # at least one non-space character
        path = SLASH + path_segment                        # /<segment>
        optional_path = pynini.closure(path, 0, 1)         # optional path

        domain_with_opt_path = domain_core + optional_path

        domain_graph_with_class_tags = (
            pynutil.insert('domain: "') + domain_with_opt_path.optimize() + pynutil.insert('"')
        )

        # ---------- protocol ----------
        protocol_symbols = pynini.closure((graph_symbols | pynini.cross(":", "colon")) + pynutil.insert(NEMO_SPACE))
        protocol_start = (pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")) + (
            pynini.accep("://") @ protocol_symbols
        )
        protocol_file_start = pynini.accep("file") + insert_space + (pynini.accep(":///") @ protocol_symbols)
        protocol_end = pynutil.add_weight(pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols, -1000)
        protocol = protocol_file_start | protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')

        # ---------- Combine all graphs ----------
        graph = pynini.Fst()  # empty

        # (1) Email pattern
        email_guard = NEMO_SIGMA + AT + NEMO_SIGMA + DOT + NEMO_SIGMA
        graph |= pynini.compose(email_guard, username + domain_graph_with_class_tags)

        # (2) Domain only (without protocol)
        # Exclude '$' (conflict with money FST) and '@' (email)
        dollar_accep = pynini.accep("$")
        excluded_symbols = DOT | dollar_accep | AT
        filtered_symbols = pynini.difference(accepted_symbols, excluded_symbols)
        accepted_characters = ASCII_ALNUM | filtered_symbols
        # Domain core graph
        graph_domain = (pynutil.insert('domain: "') + domain_core + pynutil.insert('"')).optimize()
        graph |= graph_domain

        # (3) URL with protocol
        graph |= protocol + insert_space + domain_graph_with_class_tags

        # (4) Credit card pattern: cue + 4–16 digits
        if deterministic:
            cc_digits = pynini.closure(NEMO_DIGIT, 4, 16)
            cc_phrases = (
                pynutil.insert('protocol: "')
                + cc_cues
                + pynutil.insert('" domain: "')
                + delete_space
                + cc_digits
                + pynutil.insert('"')
            )
            graph |= cc_phrases

            four = pynini.closure(NEMO_DIGIT, 4, 4)
            sep_token = pynini.union(HYPHEN, NEMO_SPACE)
            sep_del = pynutil.delete(pynini.closure(sep_token, 1))  # allow mix of - or space

            cc16_grouped = four + sep_del + four + sep_del + four + sep_del + four

            cc16_no_cue = (
                pynutil.insert('protocol: "신용카드 " ')
                + pynutil.insert('domain: "')
                + cc16_grouped
                + pynutil.insert('"')
            )

            # Give it higher priority over Date FST
            cc16_no_cue = pynutil.add_weight(cc16_no_cue.optimize(), -1.0)

            graph |= cc16_no_cue

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
