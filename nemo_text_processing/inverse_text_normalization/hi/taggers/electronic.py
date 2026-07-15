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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    DIGIT_GLYPH_TO_ASCII,
    DIGIT_WORD_TO_DEVANAGARI,
    GraphFst,
    delete_space,
    load_symbols,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, TO_LOWER, TO_UPPER


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic expressions in Hindi
    inverse text normalization: converts spoken Hindi words into written
    electronic forms such as email addresses, URLs, file paths, and domains.

        e-mail:
        e.g. कुमार एट जीमेल डॉट कॉम
             -> tokens { electronic { username: "kumar" domain: "gmail.com" } }
        URL:
        e.g. एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश गूगल डॉट कॉम
             -> tokens { electronic { domain: "https://google.com" } }
        file path (Windows):
        e.g. सी कोलन बैकवर्ड स्लैश यूजर्स बैकवर्ड स्लैश एच पी बैकवर्ड स्लैश डेस्कटॉप
             -> tokens { electronic { path: "C:\\Users\\HP\\Desktop" } }
        file path (Unix/Linux):
        e.g. फॉरवर्ड स्लैश होम फॉरवर्ड स्लैश यूजर फॉरवर्ड स्लैश डॉक्युमेंट्स
             -> tokens { electronic { path: "/home/user/documents" } }

    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        def seq(atom):
            return atom + pynini.closure(delete_space + atom)

        digit_glyphs = DIGIT_GLYPH_TO_ASCII
        digit_words = (DIGIT_WORD_TO_DEVANAGARI @ digit_glyphs).optimize()
        digit_seq = (digit_glyphs + pynini.closure(digit_glyphs)) | seq(digit_words)

        letter_map_lower = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        sym = load_symbols(get_abs_path("data/electronic/symbols.tsv"))
        spaced = {name: delete_space + fst + delete_space for name, fst in sym.items()}

        letter_map_upper = (letter_map_lower @ TO_UPPER).optimize()

        to_lower = pynini.closure(TO_LOWER | pynini.project(TO_LOWER, "output"))
        common_map_lower = (common_map @ to_lower).optimize()

        latin_run = pynini.closure(NEMO_ALPHA, 1)
        latin_run_lower = (latin_run @ to_lower).optimize()

        single_token = server_map | common_map | letter_map_lower
        token_seq = seq(single_token)

        path_atom = common_map | server_map | digit_words | digit_glyphs | latin_run | letter_map_upper
        path_atom_lower = (
            common_map_lower | server_map | digit_words | digit_glyphs | latin_run_lower | letter_map_lower
        )
        unix_path_atom = sym["and"] | path_atom_lower

        file_ext = (
            spaced["dot"]
            + seq(path_atom_lower)
            + pynini.closure((spaced["dot"] | spaced["hyphen"]) + seq(path_atom_lower))
        )

        path_sep_seg = (spaced["hyphen"] | spaced["underscore"]) + seq(path_atom)
        path_segment = (
            path_atom
            + pynini.closure(
                (delete_space + path_atom)
                | path_sep_seg
                | spaced["space"]
                | spaced["openbracket"]
                | spaced["closebracket"]
            )
            + pynini.closure(file_ext, 0, 1)
        )

        unix_sep_seg = (spaced["hyphen"] | spaced["underscore"]) + seq(unix_path_atom)
        version_seg = sym["v"] + unix_path_atom + pynini.closure(spaced["dot"] + seq(unix_path_atom))
        dollar_var = spaced["dollar"] + seq(unix_path_atom)
        unix_segment = (
            (version_seg | dollar_var | unix_path_atom)
            + pynini.closure((delete_space + unix_path_atom) | unix_sep_seg)
            + pynini.closure(file_ext, 0, 1)
        )

        lit_seg = (
            unix_path_atom
            + pynini.closure((delete_space + unix_path_atom) | unix_sep_seg | spaced["lithyphen"])
            + pynini.closure(file_ext, 0, 1)
        )

        def path_graph(prefix, segment, separator):
            return (
                pynutil.insert("path: \"")
                + prefix
                + segment
                + pynini.closure(separator + segment)
                + pynini.closure(separator, 0, 1)
                + pynutil.insert("\"")
            )

        windows_path_fst = path_graph(
            letter_map_upper + delete_space + sym["colon"] + spaced["backslash"], path_segment, spaced["backslash"]
        )
        unc_path_fst = path_graph(spaced["backslash"], path_segment, spaced["backslash"])
        unix_abs_path_fst = path_graph(spaced["forwardslash"], unix_segment, spaced["forwardslash"])
        tilde_path_fst = path_graph(sym["tilde"] + spaced["forwardslash"], unix_segment, spaced["forwardslash"])
        unix_rel_path_fst = path_graph(unix_segment + spaced["forwardslash"], unix_segment, spaced["forwardslash"])
        literal_rel_path_fst = path_graph(lit_seg + spaced["litslash"], lit_seg, spaced["litslash"])

        domain_single = server_map | common_map_lower | letter_map_lower
        domain_token_seq = seq(domain_single)

        domain_label = (digit_seq + pynini.closure(delete_space + letter_map_lower)) | domain_token_seq
        domain_body = domain_label + pynini.closure(spaced["hyphen"] + domain_label)
        compound_tld = domain_map + pynini.closure(spaced["dot"] + domain_map, 0, 2)
        full_domain = pynini.closure(domain_body + spaced["dot"], 0, 4) + domain_body + spaced["dot"] + compound_tld
        full_domain_bare = pynini.closure(domain_body + spaced["dot"], 0, 4) + domain_body

        uname_atom = sym["and"] | letter_map_lower | digit_words | digit_glyphs | server_map | common_map
        uname_sep = spaced["dot"] | spaced["hyphen"] | spaced["underscore"]
        username = uname_atom + pynini.closure((uname_sep + uname_atom) | (delete_space + uname_atom))
        email_fst = (
            pynutil.insert("username: \"")
            + username
            + pynutil.insert("\"")
            + spaced["at"]
            + pynutil.insert("domain: \"")
            + domain_body
            + spaced["dot"]
            + compound_tld
            + pynutil.insert("\"")
        )

        path_atom_url = (
            (digit_seq + spaced["x"] + digit_seq)
            | (digit_seq + delete_space + letter_map_lower + delete_space + digit_seq)
            | digit_seq
            | token_seq
        )

        inline_domain_seg = (
            pynini.closure(token_seq + spaced["dot"], 0, 2)
            + token_seq
            + spaced["dot"]
            + domain_map
            + pynini.closure(spaced["dot"] + domain_map, 0, 1)
        )

        path_segment_url = (
            path_atom_url
            + pynini.closure(spaced["hyphen"] + (digit_seq | token_seq))
            + pynini.closure(spaced["underscore"] + token_seq)
            + pynini.closure(spaced["dot"] + token_seq, 0, 1)
        )

        url_seg = (spaced["dot"] + token_seq) | inline_domain_seg | path_segment_url
        www_as_path_seg = sym["www"] + spaced["dot"] + full_domain + pynini.closure(spaced["forwardslash"] + url_seg)
        slash_with_word = spaced["forwardslash"] + (url_seg | www_as_path_seg)

        hash_frag = spaced["hashtag"] + token_seq + pynini.closure(spaced["hyphen"] + token_seq)

        url_tail = (
            pynini.closure(slash_with_word)
            + pynini.closure(spaced["forwardslash"], 0, 1)
            + pynini.closure(hash_frag, 0, 1)
        )
        domain_and_path = full_domain + url_tail
        domain_and_path_bare = full_domain_bare + url_tail

        protocol_prefix = (
            (sym["https"] | sym["http"]) + delete_space + pynini.closure(sym["www"] + spaced["dot"], 0, 1)
        )

        url_fst = pynutil.insert("domain: \"") + protocol_prefix + domain_and_path + pynutil.insert("\"")
        url_fst_bare = pynutil.insert("domain: \"") + protocol_prefix + domain_and_path_bare + pynutil.insert("\"")
        www_fst = pynutil.insert("domain: \"") + sym["www"] + spaced["dot"] + domain_and_path + pynutil.insert("\"")
        www_fst_bare = (
            pynutil.insert("domain: \"") + sym["www"] + spaced["dot"] + domain_and_path_bare + pynutil.insert("\"")
        )
        plain_fst = pynutil.insert("domain: \"") + domain_and_path + pynutil.insert("\"")

        graph = (
            email_fst
            | windows_path_fst
            | unc_path_fst
            | url_fst
            | www_fst
            | url_fst_bare
            | www_fst_bare
            | unix_abs_path_fst
            | tilde_path_fst
            | unix_rel_path_fst
            | literal_rel_path_fst
            | plain_fst
        )

        self.fst = self.add_tokens(graph).optimize()
