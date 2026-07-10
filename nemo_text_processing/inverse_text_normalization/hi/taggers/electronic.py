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
    delete_zero_or_one_space,
    load_symbols,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, TO_LOWER, TO_UPPER


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic expressions in Hindi
    inverse text normalization: converts spoken Hindi words into written
    electronic forms such as email addresses, URLs, file paths, IP addresses,
    domains, and chemical formulas.

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
        IP address:
        e.g. एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक
             -> tokens { electronic { domain: "192.168.1.1" } }
        chemical formula:
        e.g. एन ए ओ एच
             -> tokens { electronic { domain: "NaOH" } }

    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        digit_glyphs = DIGIT_GLYPH_TO_ASCII
        digit_words = (DIGIT_WORD_TO_DEVANAGARI @ digit_glyphs).optimize()
        single_digit = digit_glyphs | digit_words
        digit_seq = (digit_glyphs + pynini.closure(digit_glyphs)) | (
            digit_words + pynini.closure(delete_space + digit_words)
        )

        letter_map_lower = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        sym = load_symbols(get_abs_path("data/electronic/symbols.tsv"))

        letter_map_upper = (letter_map_lower @ TO_UPPER).optimize()

        to_lower = pynini.closure(TO_LOWER | pynini.project(TO_LOWER, "output"))
        common_map_lower = (common_map @ to_lower).optimize()

        latin_run = pynini.closure(NEMO_ALPHA, 1)
        latin_run_lower = (latin_run @ to_lower).optimize()

        drive_letter = letter_map_upper

        # One set of spacing rules applied to every spoken symbol, instead of
        # repeating the delete_space patterns individually per symbol.
        def sym_between(name):  # optional spaces on both sides
            return delete_space + sym[name] + delete_space

        def sym_before(name):  # symbol ends a segment (space before it)
            return delete_space + sym[name]

        def sym_after(name):  # symbol starts a segment (space after it)
            return sym[name] + delete_space

        def sym_soft_end(name):  # symbol with an optional single trailing space
            return delete_space + sym[name] + delete_zero_or_one_space

        seg_backslash = sym_between("backslash")
        trail_backslash = sym_before("backslash")
        lead_backslash = sym_after("backslash")

        unix_seg_slash = sym_between("forwardslash")
        unix_lead_slash = sym_after("forwardslash")
        unix_trail_slash = sym_before("forwardslash")

        url_slash = sym_before("forwardslash")

        lit_slash_seg = delete_space + pynini.accep("/") + delete_space
        lit_hyphen_seg = delete_space + pynini.accep("-") + delete_space

        dot = sym_between("dot")
        dot_end_safe = sym_soft_end("dot")
        hyphen = sym_between("hyphen")
        underscore = sym_before("underscore")
        at_sign = sym_between("at")
        x_sep = sym_before("x")
        literal_space = sym_before("space")
        open_bracket = sym_soft_end("openbracket")
        close_bracket = sym_soft_end("closebracket")
        dollar_sign = sym_before("dollar")

        and_as_letters = sym["and"]
        www_token = sym["www"]
        v_prefix = sym["v"]
        tilde_delete = sym["tilde"]

        single_token = server_map | common_map | letter_map_lower
        token_seq = single_token + pynini.closure(delete_space + single_token)

        # spelled-out letters render uppercase in Windows path segments (e.g. एच पी -> HP);
        # dictionary words keep their written form (e.g. यूज़र्स -> Users)
        path_atom = common_map | server_map | digit_words | digit_glyphs | latin_run | letter_map_upper
        path_atom_lower = (
            common_map_lower | server_map | digit_words | digit_glyphs | latin_run_lower | letter_map_lower
        )
        unix_path_atom = and_as_letters | path_atom_lower

        single_ext = (
            delete_space + sym["dot"] + delete_space + path_atom_lower + pynini.closure(delete_space + path_atom_lower)
        )
        ext_hyphen = (
            delete_space
            + sym["hyphen"]
            + delete_space
            + path_atom_lower
            + pynini.closure(delete_space + path_atom_lower)
        )
        file_ext = single_ext + pynini.closure(single_ext | ext_hyphen)

        win_hyphen = delete_space + sym["hyphen"] + delete_space + path_atom + pynini.closure(delete_space + path_atom)
        win_underscore = delete_space + sym["underscore"]
        path_segment = (
            path_atom
            + pynini.closure(
                (delete_space + path_atom) | win_hyphen | win_underscore | literal_space | open_bracket | close_bracket
            )
            + pynini.closure(file_ext, 0, 1)
        )

        unix_hyphen = (
            delete_space
            + sym["hyphen"]
            + delete_space
            + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom)
        )
        unix_underscore = (
            delete_space
            + sym["underscore"]
            + delete_space
            + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom)
        )
        version_seg = (
            v_prefix
            + unix_path_atom
            + pynini.closure(
                delete_space
                + sym["dot"]
                + delete_space
                + unix_path_atom
                + pynini.closure(delete_space + unix_path_atom)
            )
        )
        dollar_var = dollar_sign + delete_space + unix_path_atom + pynini.closure(delete_space + unix_path_atom)
        unix_segment = (
            (version_seg | dollar_var | unix_path_atom)
            + pynini.closure((delete_space + unix_path_atom) | unix_hyphen | unix_underscore)
            + pynini.closure(file_ext, 0, 1)
        )

        windows_path_fst = (
            pynutil.insert("path: \"")
            + drive_letter
            + delete_space
            + sym["colon"]
            + seg_backslash
            + path_segment
            + pynini.closure(seg_backslash + path_segment)
            + pynini.closure(trail_backslash, 0, 1)
            + pynutil.insert("\"")
        )
        unc_path_fst = (
            pynutil.insert("path: \"")
            + lead_backslash
            + path_segment
            + pynini.closure(seg_backslash + path_segment)
            + pynini.closure(trail_backslash, 0, 1)
            + pynutil.insert("\"")
        )
        unix_abs_path_fst = (
            pynutil.insert("path: \"")
            + unix_lead_slash
            + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )
        unix_rel_path_fst = (
            pynutil.insert("path: \"")
            + unix_segment
            + unix_seg_slash
            + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )
        tilde_path_fst = (
            pynutil.insert("path: \"")
            + tilde_delete
            + unix_seg_slash
            + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )

        lit_seg = (
            unix_path_atom
            + pynini.closure((delete_space + unix_path_atom) | unix_hyphen | lit_hyphen_seg)
            + pynini.closure(file_ext, 0, 1)
        )
        literal_rel_path_fst = (
            pynutil.insert("path: \"")
            + lit_seg
            + lit_slash_seg
            + lit_seg
            + pynini.closure(lit_slash_seg + lit_seg)
            + pynini.closure(pynini.cross(" /", "/"), 0, 1)
            + pynutil.insert("\"")
        )

        domain_single = server_map | common_map_lower | letter_map_lower
        domain_token_seq = domain_single + pynini.closure(delete_space + domain_single)

        digit_then_letter = digit_seq + pynini.closure(delete_space + letter_map_lower)

        first_label = (digit_seq + delete_space + letter_map_lower) | digit_seq | domain_token_seq
        domain_body = first_label + pynini.closure(hyphen + (digit_then_letter | digit_seq | domain_token_seq))
        compound_tld = domain_map + pynini.closure(dot_end_safe + domain_map, 0, 2)
        full_domain = pynini.closure(domain_body + dot, 0, 4) + domain_body + dot + compound_tld
        full_domain_bare = pynini.closure(domain_body + dot, 0, 4) + domain_body

        uname_atom = and_as_letters | letter_map_lower | digit_words | digit_glyphs | server_map | common_map
        uname_sep = (
            (delete_space + sym["dot"] + delete_space)
            | (delete_space + sym["hyphen"] + delete_space)
            | (delete_space + sym["underscore"])
        )
        username = uname_atom + pynini.closure((uname_sep + uname_atom) | (delete_space + uname_atom))
        email_fst = (
            pynutil.insert("username: \"")
            + username
            + pynutil.insert("\"")
            + at_sign
            + pynutil.insert("domain: \"")
            + domain_body
            + dot
            + compound_tld
            + pynutil.insert("\"")
        )

        ip_octet = single_digit + pynini.closure(delete_space + single_digit, 0, 2)
        ip_fst = (
            pynutil.insert("domain: \"")
            + ip_octet
            + dot
            + ip_octet
            + dot
            + ip_octet
            + dot
            + ip_octet
            + pynutil.insert("\"")
        )

        path_atom_url = (
            (digit_seq + x_sep + delete_space + digit_seq)
            | (digit_seq + delete_space + letter_map_lower + delete_space + digit_seq)
            | digit_seq
            | token_seq
        )

        inline_domain_seg = (
            pynini.closure(token_seq + dot, 0, 2)
            + token_seq
            + dot
            + domain_map
            + pynini.closure(dot + domain_map, 0, 1)
        )

        path_segment_url = (
            path_atom_url
            + pynini.closure(hyphen + (digit_seq | token_seq))
            + pynini.closure(underscore + token_seq)
            + pynini.closure(dot + token_seq, 0, 1)
        )

        slash_with_word = url_slash + ((sym["dot"] + delete_space + token_seq) | inline_domain_seg | path_segment_url)

        www_as_path_seg = www_token + dot + full_domain + pynini.closure(slash_with_word)

        slash_with_word = url_slash + (
            (delete_space + digit_seq + x_sep + delete_space + digit_seq)
            | (delete_space + digit_seq)
            | (delete_space + sym["dot"] + delete_space + token_seq)
            | (delete_space + inline_domain_seg)
            | (delete_space + www_as_path_seg)
            | (delete_space + path_segment_url)
        )

        hash_frag_body = token_seq + pynini.closure(hyphen + token_seq)
        hash_frag = delete_space + sym["hashtag"] + delete_space + hash_frag_body

        domain_and_path = (
            full_domain
            + pynini.closure(slash_with_word)
            + pynini.closure(url_slash, 0, 1)
            + pynini.closure(hash_frag, 0, 1)
        )
        domain_and_path_bare = (
            full_domain_bare
            + pynini.closure(slash_with_word)
            + pynini.closure(url_slash, 0, 1)
            + pynini.closure(hash_frag, 0, 1)
        )

        protocol = sym["https"] | sym["http"]

        url_fst = (
            pynutil.insert("domain: \"")
            + protocol
            + delete_space
            + pynini.closure(www_token + dot, 0, 1)
            + domain_and_path
            + pynutil.insert("\"")
        )
        www_fst = pynutil.insert("domain: \"") + www_token + dot + domain_and_path + pynutil.insert("\"")
        plain_fst = pynutil.insert("domain: \"") + domain_and_path + pynutil.insert("\"")

        url_fst_bare = (
            pynutil.insert("domain: \"")
            + protocol
            + delete_space
            + pynini.closure(www_token + dot, 0, 1)
            + domain_and_path_bare
            + pynutil.insert("\"")
        )
        www_fst_bare = pynutil.insert("domain: \"") + www_token + dot + domain_and_path_bare + pynutil.insert("\"")

        chem_token = digit_glyphs | letter_map_lower | letter_map_upper
        chem_more = pynini.closure(
            (delete_space + chem_token)
            | open_bracket
            | close_bracket
            | (delete_space + sym["chemopen"])
            | (delete_space + sym["chemclose"])
            | (delete_space + sym["minus"])
        )

        chem_spelled_fst = (
            pynutil.insert("domain: \"") + (chem_token + delete_space + chem_token + chem_more) + pynutil.insert("\"")
        )

        graph = (
            ip_fst
            | email_fst
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
            | chem_spelled_fst
            | plain_fst
        )

        self.fst = self.add_tokens(graph).optimize()
