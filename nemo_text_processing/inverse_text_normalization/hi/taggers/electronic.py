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
    DEVANAGARI_DIGIT,
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
    load_symbols,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import TO_LOWER, TO_UPPER


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

        digit_glyphs = pynini.union(
            *[pynini.cross(glyph, str(value)) for value, glyph in enumerate(DEVANAGARI_DIGIT)]
        ).optimize()
        cardinal_digit_words = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        )
        digit_words = (cardinal_digit_words @ digit_glyphs).optimize()
        single_digit = pynutil.add_weight(digit_glyphs, 0.8) | pynutil.add_weight(digit_words, 0.50)
        digit_seq = pynutil.add_weight(digit_glyphs + pynini.closure(digit_glyphs, 0), 0.8) | pynutil.add_weight(
            digit_words + pynini.closure(delete_space + digit_words, 0), 0.9
        )

        letter_map_lower = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        sym = load_symbols(get_abs_path("data/electronic/symbols.tsv"))

        letter_map_upper = (letter_map_lower @ TO_UPPER).optimize()
        letter_map = letter_map_lower | letter_map_upper

        to_lower = pynini.closure(TO_LOWER | pynini.project(TO_LOWER, "output"))
        common_map_lower = (common_map @ to_lower).optimize()

        latin_run = pynini.closure(
            pynini.union(*[pynini.accep(chr(c)) for c in range(ord('A'), ord('Z') + 1)])
            | pynini.union(*[pynini.accep(chr(c)) for c in range(ord('a'), ord('z') + 1)]),
            1,
        )
        latin_run_lower = (latin_run @ to_lower).optimize()

        drive_letter = letter_map_upper

        seg_backslash = delete_space + sym["backslash"] + delete_space
        trail_backslash = delete_space + sym["backslash"]
        lead_backslash = sym["backslash"] + delete_space

        unix_seg_slash = delete_space + sym["forwardslash"] + delete_space
        unix_lead_slash = sym["forwardslash"] + delete_space
        unix_trail_slash = delete_space + sym["forwardslash"]

        url_slash = delete_space + sym["forwardslash"]

        lit_slash_seg = pynini.cross(" / ", "/")
        lit_hyphen_seg = pynini.cross(" - ", "-")

        dot = delete_space + sym["dot"] + delete_space
        dot_end_safe = delete_space + sym["dot"] + delete_zero_or_one_space
        hyphen = delete_space + sym["hyphen"] + delete_space
        underscore = delete_space + sym["underscore"]
        at_sign = delete_space + sym["at"] + delete_space
        x_sep = delete_space + sym["x"]
        literal_space = delete_space + sym["space"]
        open_bracket = delete_space + sym["openbracket"] + delete_zero_or_one_space
        close_bracket = delete_space + sym["closebracket"] + delete_zero_or_one_space
        dollar_sign = delete_space + sym["dollar"]

        lit_open_paren = delete_space + pynutil.delete("(") + pynutil.insert("(") + delete_zero_or_one_space
        lit_close_paren = delete_space + pynutil.delete(")") + pynutil.insert(")")
        or_word = sym["or"]
        and_as_letters = sym["and"]
        www_token = sym["www"]
        v_prefix = sym["v"]
        hp_token = sym["hp"]
        tilde_delete = sym["tilde"]

        single_token = pynutil.add_weight(server_map | common_map, 0.90) | pynutil.add_weight(letter_map_lower, 1.00)
        token_seq = single_token + pynini.closure(delete_space + single_token, 0)

        path_atom = pynutil.add_weight(
            hp_token | www_token | or_word | and_as_letters | common_map | server_map, 0.90
        ) | pynutil.add_weight(digit_words | digit_glyphs | latin_run | letter_map, 1.00)
        path_atom_lower = pynutil.add_weight(common_map_lower | server_map, 0.90) | pynutil.add_weight(
            digit_words | digit_glyphs | latin_run_lower | letter_map_lower, 1.00
        )
        unix_path_atom = (
            pynutil.add_weight(www_token | or_word | and_as_letters, 0.84)
            | pynutil.add_weight(pynini.cross("CI", "c"), 0.86)
            | path_atom_lower
        )

        single_ext = (
            delete_space
            + sym["dot"]
            + delete_space
            + path_atom_lower
            + pynini.closure(delete_space + path_atom_lower, 0)
        )
        ext_hyphen = (
            delete_space
            + sym["hyphen"]
            + delete_space
            + path_atom_lower
            + pynini.closure(delete_space + path_atom_lower, 0)
        )
        file_ext = single_ext + pynini.closure(single_ext | ext_hyphen, 0)

        win_hyphen = (
            delete_space + sym["hyphen"] + delete_space + path_atom + pynini.closure(delete_space + path_atom, 0)
        )
        win_underscore = delete_space + sym["underscore"]
        path_segment = (
            path_atom
            + pynini.closure(
                pynutil.add_weight(
                    (delete_space + path_atom)
                    | win_hyphen
                    | win_underscore
                    | literal_space
                    | open_bracket
                    | close_bracket
                    | lit_open_paren
                    | lit_close_paren,
                    1.0,
                ),
                0,
            )
            + pynini.closure(file_ext, 0, 1)
        )

        unix_hyphen = (
            delete_space
            + sym["hyphen"]
            + delete_space
            + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom, 0)
        )
        unix_underscore = (
            delete_space
            + sym["underscore"]
            + delete_space
            + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom, 0)
        )
        version_seg = (
            v_prefix
            + unix_path_atom
            + pynini.closure(
                delete_space
                + sym["dot"]
                + delete_space
                + unix_path_atom
                + pynini.closure(delete_space + unix_path_atom, 0),
                0,
            )
        )
        dollar_var = dollar_sign + delete_space + unix_path_atom + pynini.closure(delete_space + unix_path_atom, 0)
        unix_segment = (
            (
                pynutil.add_weight(version_seg, 0.85)
                | pynutil.add_weight(dollar_var, 0.87)
                | pynutil.add_weight(unix_path_atom, 1.00)
            )
            + pynini.closure(
                pynutil.add_weight(
                    (delete_space + unix_path_atom) | unix_hyphen | unix_underscore,
                    1.0,
                ),
                0,
            )
            + pynini.closure(file_ext, 0, 1)
        )

        windows_path_fst = (
            pynutil.insert("path: \"")
            + drive_letter
            + delete_space
            + sym["colon"]
            + seg_backslash
            + path_segment
            + pynini.closure(seg_backslash + path_segment, 0)
            + pynini.closure(trail_backslash, 0, 1)
            + pynutil.insert("\"")
        )
        unc_path_fst = (
            pynutil.insert("path: \"")
            + lead_backslash
            + path_segment
            + pynini.closure(seg_backslash + path_segment, 0)
            + pynini.closure(trail_backslash, 0, 1)
            + pynutil.insert("\"")
        )
        unix_abs_path_fst = (
            pynutil.insert("path: \"")
            + unix_lead_slash
            + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment, 0)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )
        unix_rel_path_fst = (
            pynutil.insert("path: \"")
            + unix_segment
            + unix_seg_slash
            + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment, 0)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )
        tilde_path_fst = (
            pynutil.insert("path: \"")
            + tilde_delete
            + unix_seg_slash
            + unix_segment
            + pynini.closure(unix_seg_slash + unix_segment, 0)
            + pynini.closure(unix_trail_slash, 0, 1)
            + pynutil.insert("\"")
        )

        lit_seg = (
            unix_path_atom
            + pynini.closure(
                pynutil.add_weight(
                    (delete_space + unix_path_atom) | unix_hyphen | lit_hyphen_seg,
                    1.0,
                ),
                0,
            )
            + pynini.closure(file_ext, 0, 1)
        )
        literal_rel_path_fst = (
            pynutil.insert("path: \"")
            + lit_seg
            + lit_slash_seg
            + lit_seg
            + pynini.closure(lit_slash_seg + lit_seg, 0)
            + pynini.closure(pynini.cross(" /", "/"), 0, 1)
            + pynutil.insert("\"")
        )

        domain_single = pynutil.add_weight(server_map | common_map_lower, 0.90) | pynutil.add_weight(
            letter_map_lower, 1.00
        )
        domain_token_seq = domain_single + pynini.closure(delete_space + domain_single, 0)

        digit_then_letter = digit_seq + pynini.closure(delete_space + letter_map_lower, 0)

        first_label = (
            pynutil.add_weight(digit_seq + delete_space + letter_map_lower, 0.85)
            | pynutil.add_weight(digit_seq, 0.87)
            | pynutil.add_weight(domain_token_seq, 1.00)
        )
        domain_body = first_label + pynini.closure(
            hyphen
            + (pynutil.add_weight(digit_then_letter, 0.85) | pynutil.add_weight(digit_seq | domain_token_seq, 1.0)),
            0,
        )
        compound_tld = domain_map + pynini.closure(dot_end_safe + domain_map, 0, 2)
        full_domain = pynini.closure(domain_body + dot, 0, 4) + domain_body + dot + compound_tld
        full_domain_bare = pynini.closure(domain_body + dot, 0, 4) + domain_body

        uname_atom = (
            pynutil.add_weight(and_as_letters | letter_map_lower, 0.84)
            | pynutil.add_weight(digit_words | digit_glyphs, 0.88)
            | pynutil.add_weight(server_map, 0.90)
            | pynutil.add_weight(common_map, 0.95)
        )
        uname_sep = (
            (delete_space + sym["dot"] + delete_space)
            | (delete_space + sym["hyphen"] + delete_space)
            | (delete_space + sym["underscore"])
        )
        username = uname_atom + pynini.closure((uname_sep + uname_atom) | (delete_space + uname_atom), 0)
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
            pynutil.add_weight(digit_seq + x_sep + delete_space + digit_seq, 0.75)
            | pynutil.add_weight(digit_seq + delete_space + letter_map_lower + delete_space + digit_seq, 0.80)
            | pynutil.add_weight(digit_words, 0.88)
            | pynutil.add_weight(digit_glyphs, 0.89)
            | pynutil.add_weight(digit_seq, 0.90)
            | pynutil.add_weight(token_seq, 1.00)
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
            + pynini.closure(hyphen + (digit_seq | token_seq), 0)
            + pynini.closure(underscore + token_seq, 0)
            + pynini.closure(dot + token_seq, 0, 1)
        )

        slash_with_word = url_slash + (
            pynutil.add_weight(sym["dot"] + delete_space + token_seq, 0.90)
            | pynutil.add_weight(inline_domain_seg, 0.95)
            | pynutil.add_weight(path_segment_url, 1.00)
        )

        www_as_path_seg = www_token + dot + full_domain + pynini.closure(slash_with_word, 0)

        slash_with_word = url_slash + (
            pynutil.add_weight(delete_space + digit_seq + x_sep + delete_space + digit_seq, 0.20)
            | pynutil.add_weight(delete_space + digit_seq, 0.30)
            | pynutil.add_weight(delete_space + sym["dot"] + delete_space + token_seq, 0.90)
            | pynutil.add_weight(delete_space + inline_domain_seg, 0.95)
            | pynutil.add_weight(delete_space + www_as_path_seg, 0.97)
            | pynutil.add_weight(delete_space + path_segment_url, 1.00)
        )

        hash_frag_body = token_seq + pynini.closure(hyphen + token_seq, 0)
        hash_frag = delete_space + sym["hashtag"] + delete_space + hash_frag_body

        domain_and_path = (
            full_domain
            + pynini.closure(slash_with_word, 0)
            + pynini.closure(url_slash, 0, 1)
            + pynini.closure(hash_frag, 0, 1)
        )
        domain_and_path_bare = (
            full_domain_bare
            + pynini.closure(slash_with_word, 0)
            + pynini.closure(url_slash, 0, 1)
            + pynini.closure(hash_frag, 0, 1)
        )

        protocol = pynutil.add_weight(sym["https"] | sym["http"], 1.0)

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

        chem_token = pynutil.add_weight(digit_glyphs, 0.90) | pynutil.add_weight(letter_map, 1.00)
        chem_more = pynini.closure(
            pynutil.add_weight(
                (delete_space + chem_token)
                | open_bracket
                | close_bracket
                | (delete_space + sym["chemopen"])
                | (delete_space + sym["chemclose"])
                | (delete_space + sym["minus"]),
                1.0,
            ),
            0,
        )

        chem_spelled_fst = (
            pynutil.insert("domain: \"") + (chem_token + delete_space + chem_token + chem_more) + pynutil.insert("\"")
        )

        graph = (
            pynutil.add_weight(ip_fst, 1.00)
            | pynutil.add_weight(email_fst, 1.05)
            | pynutil.add_weight(windows_path_fst, 1.06)
            | pynutil.add_weight(unc_path_fst, 1.07)
            | pynutil.add_weight(url_fst, 0.10)
            | pynutil.add_weight(www_fst, 0.11)
            | pynutil.add_weight(url_fst_bare, 0.50)
            | pynutil.add_weight(www_fst_bare, 0.51)
            | pynutil.add_weight(unix_abs_path_fst, 15.00)
            | pynutil.add_weight(tilde_path_fst, 1.12)
            | pynutil.add_weight(unix_rel_path_fst, 15.00)
            | pynutil.add_weight(literal_rel_path_fst, 1.15)
            | pynutil.add_weight(chem_spelled_fst, 1.18)
            | pynutil.add_weight(plain_fst, 1.30)
        )

        self.fst = self.add_tokens(graph).optimize()
