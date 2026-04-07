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
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
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
        electronic { username: "kumar" domain: "gmail.com" } -> "कुमार एट जीमेल डॉट कॉम"
        electronic { protocol: "https" domain: "google.com/" } -> "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश गूगल डॉट कॉम फॉरवर्ड स्लैश"
        electronic { path: "C:\\Users\\HP" } -> "सी कोलन बैकवर्ड स्लैश यूज़र्स बैकवर्ड स्लैश एच पी"
        electronic { ip: "192.168.1.1" } -> "एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        # Load data files
        symbols_graph = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        domain_graph = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).optimize()
        server_name_graph = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).optimize()
        common_words_graph = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).optimize()
        latin_to_hindi_graph = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        latin_to_hindi_graph = capitalized_input_graph(latin_to_hindi_graph).optimize()

        # Digit mappings - use telephone number mappings for ASCII digits
        ascii_digit_graph = pynini.string_file(get_abs_path("data/telephone/number.tsv")).optimize()
        hindi_digit_graph = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        hindi_zero_graph = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
        subscript_digit_graph = pynini.string_file(get_abs_path("data/electronic/subscript_digit.tsv")).optimize()
        digit_verbalization = ascii_digit_graph | hindi_digit_graph | hindi_zero_graph | subscript_digit_graph

        # Combined phonetic word graph: server names + common words
        phonetic_word = server_name_graph | common_words_graph

        # ============ CHARACTER VERBALIZATION ============
        # Single character to Hindi verbalization with space insertion
        char_to_hindi = pynutil.add_weight(latin_to_hindi_graph, 1.0) | pynutil.add_weight(  # Letter mapping
            digit_verbalization, 1.0
        )  # Digit mapping
        char_with_space = char_to_hindi + insert_space

        # ============ SYMBOL VERBALIZATION ============
        symbol_to_hindi = symbols_graph + insert_space

        # ============ WORD SEGMENTATION & VERBALIZATION ============
        # Try to match complete words first, then fall back to letter-by-letter
        word_char = NEMO_ALPHA | NEMO_DIGIT

        # For a sequence of word characters, try phonetic first, else letter-by-letter
        word_segment = pynini.closure(word_char, 1)

        # Phonetic word verbalization (higher priority)
        phonetic_verbalization = phonetic_word + insert_space

        # Letter-by-letter verbalization (fallback)
        letter_by_letter = pynini.closure(char_with_space, 1)

        # Combined: try phonetic first, fall back to letter-by-letter
        # This is done by using weights - phonetic has lower weight (higher priority)
        word_verbalization = pynutil.add_weight(phonetic_verbalization, 0.9) | pynutil.add_weight(
            letter_by_letter, 1.1
        )

        # ============ DOMAIN VERBALIZATION ============
        # Domain extension verbalization (.com -> डॉट कॉम)
        domain_ext_verbalization = pynini.cross(".", "डॉट ") + domain_graph + insert_space

        # ============ PROTOCOL VERBALIZATION ============
        protocol_graph = pynini.string_file(get_abs_path("data/electronic/protocols.tsv")).optimize()
        protocol_verbalization = protocol_graph + insert_space

        # ============ CONTENT VERBALIZATION ============
        # General content: mix of words, symbols, and characters
        # Process character by character with symbol handling
        content_char = pynutil.add_weight(symbol_to_hindi, 1.0) | pynutil.add_weight(  # Symbol
            char_with_space, 1.1
        )  # Single char

        # Full content verbalization
        content_verbalization = pynini.closure(content_char, 1)

        # ============ FIELD EXTRACTION ============
        # Extract username field
        delete_username_tag = pynutil.delete("username: \"")
        delete_domain_tag = pynutil.delete("domain: \"")
        delete_protocol_tag = pynutil.delete("protocol: \"")
        delete_path_tag = pynutil.delete("path: \"")
        delete_ip_tag = pynutil.delete("ip: \"")
        delete_quote = pynutil.delete("\"")

        # Username verbalization: letter-by-letter with symbol handling
        username_content = pynini.closure(
            pynutil.add_weight(phonetic_word + insert_space, 0.9)
            | pynutil.add_weight(symbol_to_hindi, 1.0)
            | pynutil.add_weight(char_with_space, 1.1),
            1,
        )

        username_graph = (
            delete_username_tag + username_content + delete_quote + delete_space + pynutil.insert("एट ")  # @ symbol
        )

        # Domain verbalization
        domain_content = pynini.closure(
            pynutil.add_weight(phonetic_word + insert_space, 0.9)
            | pynutil.add_weight(domain_ext_verbalization, 0.95)
            | pynutil.add_weight(symbol_to_hindi, 1.0)
            | pynutil.add_weight(char_with_space, 1.1),
            1,
        )

        domain_only_graph = delete_domain_tag + domain_content + delete_quote

        # Protocol verbalization
        protocol_only_graph = delete_protocol_tag + protocol_verbalization + delete_quote + delete_space

        # Path verbalization (Windows/Unix file paths)
        path_content = pynini.closure(
            pynutil.add_weight(common_words_graph + insert_space, 0.9)
            | pynutil.add_weight(symbol_to_hindi, 1.0)
            | pynutil.add_weight(char_with_space, 1.1),
            1,
        )

        path_graph = delete_path_tag + path_content + delete_quote

        # IP address verbalization (digit by digit)
        ip_char = pynutil.add_weight(symbols_graph + insert_space, 1.0) | pynutil.add_weight(
            digit_verbalization + insert_space, 1.0
        )
        ip_content = pynini.closure(ip_char, 1)

        ip_graph = delete_domain_tag + ip_content + delete_quote

        # ============ COMBINED GRAPH ============
        # Email: username + domain
        email_full = username_graph + domain_only_graph

        # URL with protocol: protocol + domain
        url_full = protocol_only_graph + domain_only_graph

        # Combined final graph
        graph = (
            pynutil.add_weight(url_full, 1.0)
            | pynutil.add_weight(email_full, 1.01)
            | pynutil.add_weight(path_graph, 1.02)
            | pynutil.add_weight(ip_graph, 1.03)
            | pynutil.add_weight(domain_only_graph, 1.04)
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
