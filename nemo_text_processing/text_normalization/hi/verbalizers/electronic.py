# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    GraphFst,
    capitalized_input_graph,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic addresses.
    Uses a phonetic-first approach with letter-by-letter fallback.

    Examples:
        electronic { username: "kumar" domain: "gmail.com" } -> "के यू एम ए आर एट जीमेल डॉट कॉम"
        electronic { protocol: "https" domain: "google.com/" } -> "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश गूगल डॉट कॉम फॉरवर्ड स्लैश"
        electronic { path: "C:\\Users\\HP\\Desktop" } -> "सी कोलन बैकवर्ड स्लैश यूज़र्स बैकवर्ड स्लैश एच पी बैकवर्ड स्लैश डेस्कटॉप"
        electronic { domain: "192.168.1.1" } -> "एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        symbols_graph         = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        domain_graph          = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).optimize()
        server_name_graph     = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).optimize()
        chemical_graph        = pynini.string_file(get_abs_path("data/electronic/chemical_names.tsv")).optimize()
        common_words_graph    = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).optimize()
        latin_to_hindi_graph  = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        latin_to_hindi_graph  = capitalized_input_graph(latin_to_hindi_graph).optimize()

        ascii_digit_graph     = pynini.string_file(get_abs_path("data/telephone/number.tsv")).optimize()
        hindi_digit_graph     = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        hindi_zero_graph      = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
        subscript_digit_graph = pynini.string_file(get_abs_path("data/electronic/subscript_digit.tsv")).optimize()
        digit_verbalization   = (
            ascii_digit_graph | hindi_digit_graph | hindi_zero_graph | subscript_digit_graph
        )

        protocol_graph = pynini.string_file(get_abs_path("data/electronic/protocols.tsv")).optimize()

        single_letter = latin_to_hindi_graph + insert_space
        single_digit  = digit_verbalization  + insert_space
        single_symbol = symbols_graph        + insert_space

        single_non_alpha = (
            pynutil.add_weight(single_symbol, 1.0)
            | pynutil.add_weight(single_digit,  1.0)
        )

        def make_alpha_run_verbalizer(tsv_graphs):
            phonetic = pynini.union(
                *[pynutil.add_weight(g + insert_space, w) for g, w in tsv_graphs]
            )
            literal = pynutil.add_weight(pynini.closure(single_letter, 1), 1.1)
            return phonetic | literal

        def make_content(alpha_run_verb, non_alpha_sep=None):
            if non_alpha_sep is None:
                non_alpha_sep = single_non_alpha
            mandatory_sep = pynini.closure(non_alpha_sep, 1)
            return (
                pynini.closure(non_alpha_sep, 0)
                + pynini.closure(alpha_run_verb + mandatory_sep, 0)
                + pynini.closure(alpha_run_verb, 0, 1)
                + pynini.closure(non_alpha_sep, 0)
            )

        delete_username_tag = pynutil.delete("username: \"")
        delete_domain_tag   = pynutil.delete("domain: \"")
        delete_protocol_tag = pynutil.delete("protocol: \"")
        delete_path_tag     = pynutil.delete("path: \"")
        delete_quote        = pynutil.delete("\"")

        username_alpha_run = make_alpha_run_verbalizer([
            (server_name_graph,  0.85),
            (domain_graph,       0.87),   
            (common_words_graph, 0.90),
        ])
        username_content = make_content(username_alpha_run)
        username_graph = (
            delete_username_tag
            + username_content
            + delete_quote
            + delete_space
            + pynutil.insert("एट ")
        )

        domain_alpha_run = make_alpha_run_verbalizer([
            (server_name_graph,  0.85),
            (domain_graph,       0.87),
            (common_words_graph, 0.90),
        ])
        
        domain_content = (
            pynutil.add_weight(chemical_graph + insert_space, 0.8)
            | pynutil.add_weight(make_content(domain_alpha_run), 1.0)
        )
        
        domain_only_graph = delete_domain_tag + domain_content + delete_quote

        protocol_only_graph = (
            delete_protocol_tag
            + protocol_graph + insert_space
            + delete_quote
            + delete_space
        )

        path_alpha_run = make_alpha_run_verbalizer([
            (domain_graph,       0.87),
            (common_words_graph, 0.90),
        ])
        path_content = make_content(path_alpha_run)
        path_graph   = delete_path_tag + path_content + delete_quote

        ip_char    = single_symbol | single_digit
        ip_content = pynini.closure(ip_char, 1)
        ip_graph   = delete_domain_tag + ip_content + delete_quote

        email_full = username_graph + domain_only_graph
        url_full   = protocol_only_graph + domain_only_graph

        graph = (
            pynutil.add_weight(url_full,           1.0)
            | pynutil.add_weight(email_full,        1.01)
            | pynutil.add_weight(path_graph,        1.02)
            | pynutil.add_weight(ip_graph,          1.03)
            | pynutil.add_weight(domain_only_graph, 1.04)
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()