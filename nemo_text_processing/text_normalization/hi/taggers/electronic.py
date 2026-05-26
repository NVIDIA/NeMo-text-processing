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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_ALPHA, NEMO_DIGIT, NEMO_HI_DIGIT, GraphFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, file paths,
    IP addresses, domains, chemical formulas, and alphanumeric codes.
        e.g. kumar@gmail.com -> tokens { electronic { username: "kumar" domain: "gmail.com" } }
        e.g. https://google.com/ -> tokens { electronic { protocol: "https" domain: "google.com/" } }
        e.g. C:\\Users\\HP\\Desktop -> tokens { electronic { path: "C:\\Users\\HP\\Desktop" } }
        e.g. 192.168.1.1 -> tokens { electronic { domain: "192.168.1.1" } }

    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        subscript_digit = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/subscript_digit.tsv")), "input"
        )

        alphanumeric = NEMO_ALPHA | NEMO_DIGIT | NEMO_HI_DIGIT | subscript_digit

        username_chars = NEMO_ALPHA | NEMO_DIGIT | pynini.accep(".") | pynini.accep("-") | pynini.accep("_")
        username = pynutil.insert("username: \"") + pynini.closure(username_chars, 1) + pynutil.insert("\"")

        domain_chars = NEMO_ALPHA | NEMO_DIGIT | pynini.accep(".") | pynini.accep("-")
        domain = pynutil.insert(" domain: \"") + pynini.closure(domain_chars, 1) + pynutil.insert("\"")

        email_graph = username + pynini.cross("@", "") + domain

        protocol_start = pynini.cross("https://", "https") | pynini.cross("http://", "http")
        protocol_end = pynini.cross("www.", "www")
        protocol = (
            pynutil.insert("protocol: \"")
            + (
                pynutil.add_weight(protocol_start + protocol_end, 1.0)
                | pynutil.add_weight(protocol_start, 1.01)
                | pynutil.add_weight(protocol_end, 1.02)
            )
            + pynutil.insert("\"")
        )

        url_path_chars = alphanumeric | pynini.union(
            pynini.accep("."),
            pynini.accep("-"),
            pynini.accep("_"),
            pynini.accep("/"),
            pynini.accep("#"),
            pynini.accep("?"),
            pynini.accep("&"),
            pynini.accep("="),
            pynini.accep("%"),
            pynini.accep("+"),
            pynini.accep(":"),
        )
        url_path = pynini.closure(url_path_chars, 1)

        url_domain = pynutil.insert(" domain: \"") + url_path + pynutil.insert("\"")

        url_graph = protocol + url_domain

        drive_letter = NEMO_ALPHA
        windows_path_chars = alphanumeric | pynini.union(
            pynini.accep("\\"),
            pynini.accep("."),
            pynini.accep("-"),
            pynini.accep("_"),
            pynini.accep(" "),
            pynini.accep("("),
            pynini.accep(")"),
        )
        windows_path = (
            pynutil.insert("path: \"")
            + drive_letter
            + pynini.accep(":")
            + pynini.accep("\\")
            + pynini.closure(windows_path_chars, 1)
            + pynutil.insert("\"")
        )

        unix_path_chars = alphanumeric | pynini.union(
            pynini.accep("/"),
            pynini.accep("."),
            pynini.accep("-"),
            pynini.accep("_"),
            pynini.accep("$"),
        )
        
        unix_segment_chars = alphanumeric | pynini.union(
            pynini.accep("."),
            pynini.accep("-"),
            pynini.accep("_"),
            pynini.accep("$"),
        )
        unix_segment = pynini.closure(unix_segment_chars, 1)

        abs_unix_path = pynini.accep("/") + pynini.closure(unix_path_chars, 1)
        
        rel_unix_path = unix_segment + pynini.accep("/") + pynini.closure(unix_path_chars, 0)
        
        unix_path = (
            pynutil.insert("path: \"") 
            + (abs_unix_path | rel_unix_path) 
            + pynutil.insert("\"")
        )

        backslash_path_chars = alphanumeric | pynini.union(
            pynini.accep("\\"),
            pynini.accep("."),
            pynini.accep("-"),
            pynini.accep("_"),
            pynini.accep(" "),
        )
        backslash_path = (
            pynutil.insert("path: \"")
            + pynini.accep("\\")
            + pynini.closure(backslash_path_chars, 1)
            + pynutil.insert("\"")
        )

        ip_octet = pynini.closure(NEMO_DIGIT, 1, 3)
        dot_octet = pynini.accep(".") + ip_octet
        ip_address = pynutil.insert("domain: \"") + ip_octet + pynini.closure(dot_octet, 3, 3) + pynutil.insert("\"")

        domain_segment_chars = NEMO_ALPHA | NEMO_DIGIT | pynini.accep("-")
        domain_segment = pynini.closure(domain_segment_chars, 1)

        tld = pynini.project(pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input")

        domain_body = (
            pynini.closure(domain_segment + pynini.accep("."), 1) + tld + pynini.closure(pynini.accep(".") + tld, 0, 1)
        )

        combined_domain = (
            pynutil.insert("domain: \"") + domain_body + pynini.closure(pynini.accep("/"), 0, 1) + pynutil.insert("\"")
        )

        known_extensions = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv")), "input"
        )
        filename_stem_chars = NEMO_ALPHA | NEMO_DIGIT | pynini.accep("-") | pynini.accep("_")
        filename_stem = pynini.closure(filename_stem_chars, 1)
        file_with_extension = (
            pynutil.insert("domain: \"") + filename_stem + pynini.accep(".") + known_extensions + pynutil.insert("\"")
        )

        chemical_chars = (
            NEMO_ALPHA 
            | NEMO_DIGIT 
            | subscript_digit 
            | pynini.accep("(") 
            | pynini.accep(")")
            | pynini.accep("+")
            | pynini.accep("-")
            | pynini.accep("–") 
        )
        
        raw_chemical = NEMO_ALPHA + pynini.closure(chemical_chars, 1)
        
        any_chem = pynini.closure(chemical_chars)
        has_open = any_chem + pynini.accep("(") + any_chem
        no_open = pynini.difference(any_chem, has_open)
        ends_with_close = any_chem + pynini.accep(")")
        
        unbalanced_trailing = pynini.intersect(no_open, ends_with_close)
        
        valid_chemical = pynini.difference(raw_chemical, unbalanced_trailing).optimize()
        
        chemical_formula = (
            pynutil.insert("domain: \"") 
            + valid_chemical 
            + pynutil.insert("\"")
        )
    
        alnum_seg = pynini.closure(NEMO_ALPHA | NEMO_DIGIT, 1)
        
        separator = pynini.accep("-") | pynini.accep(".")
        alphanumeric_pattern = alnum_seg + pynini.closure(separator + alnum_seg)

        alnum_hyp_dot_sigma = pynini.closure(NEMO_ALPHA | NEMO_DIGIT | pynini.accep("-") | pynini.accep("."))
        
        contains_alpha = alnum_hyp_dot_sigma + NEMO_ALPHA + alnum_hyp_dot_sigma
        contains_digit = alnum_hyp_dot_sigma + NEMO_DIGIT + alnum_hyp_dot_sigma
        
        alphanumeric_code_fst = pynini.intersect(
            pynini.intersect(alphanumeric_pattern, contains_alpha), contains_digit
        ).optimize()

        alphanumeric_code = pynutil.insert("domain: \"") + alphanumeric_code_fst + pynutil.insert("\"")

        graph = (
            pynutil.add_weight(url_graph, 1.0)
            | pynutil.add_weight(email_graph, 1.0)
            | pynutil.add_weight(windows_path, 1.0)
            | pynutil.add_weight(unix_path, 1.0)
            | pynutil.add_weight(backslash_path, 1.0)
            | pynutil.add_weight(ip_address, 1.0)
            | pynutil.add_weight(combined_domain, 1.1)
            | pynutil.add_weight(file_with_extension, 1.1)
            | pynutil.add_weight(chemical_formula, 1.2)
            | pynutil.add_weight(alphanumeric_code, 1.2)
        )

        self.graph = graph.optimize()
        self.fst = self.add_tokens(graph).optimize()