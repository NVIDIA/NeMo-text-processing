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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic expressions in Hindi
    inverse text normalization: converts spoken Hindi words into written
    electronic forms such as email addresses, URLs, file paths, IP addresses,
    domains, chemical formulas, and alphanumeric codes.

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
             -> tokens { electronic { ip: "192.168.1.1" } }
        chemical formula:
        e.g. एन ए ओ एच
             -> tokens { electronic { domain: "NaOH" } }
        alphanumeric code:
        e.g. ए बी सी एक दो तीन
             -> tokens { electronic { domain: "ABC123" } }

    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")
        digit_words = pynini.string_file(get_abs_path("data/electronic/digit_words.tsv"))
        digit_glyphs = pynini.string_file(get_abs_path("data/electronic/digit_glyphs.tsv")).invert()
        single_digit = pynutil.add_weight(digit_glyphs, 0.8) | pynutil.add_weight(digit_words, 0.50)
        digit_seq = pynutil.add_weight(digit_glyphs + pynini.closure(digit_glyphs, 0), 0.8) | pynutil.add_weight(
            digit_words + pynini.closure(delete_space + digit_words, 0), 0.9
        )

        letter_map = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        try:
            chem_named_map = pynini.string_file(get_abs_path("data/electronic/chemical_formulas.tsv")).optimize()
        except Exception:
            chem_named_map = None

        special_codes_map = pynini.string_file(get_abs_path("data/electronic/special_codes.tsv")).optimize()

        to_lower = pynini.cdrewrite(
            pynini.string_map([(chr(c), chr(c + 32)) for c in range(ord('A'), ord('Z') + 1)]),
            "",
            "",
            NEMO_SIGMA,
        )
        to_upper = pynini.cdrewrite(
            pynini.string_map([(chr(c + 32), chr(c)) for c in range(ord('A'), ord('Z') + 1)]),
            "",
            "",
            NEMO_SIGMA,
        )

        def make_lower(fst):
            return (fst @ to_lower).optimize()

        def make_upper(fst):
            return (fst @ to_upper).optimize()

        letter_map_lower = make_lower(letter_map)
        letter_map_upper = make_upper(letter_map)
        common_map_lower = make_lower(common_map)
        server_map_lower = make_lower(server_map)

        latin_run = pynini.closure(
            pynini.union(*[pynini.accep(chr(c)) for c in range(ord('A'), ord('Z') + 1)])
            | pynini.union(*[pynini.accep(chr(c)) for c in range(ord('a'), ord('z') + 1)]),
            1,
        )
        latin_run_lower = make_lower(latin_run)

        _drive_chars = pynini.union("C", "D", "E", "F", "G", "H", "I", "J")
        drive_letter = pynini.compose(letter_map_upper, _drive_chars)

        def _backslash():
            return pynutil.delete("बैकवर्ड") + delete_space + pynutil.delete("स्लैश") + pynutil.insert("\\\\")

        seg_backslash = delete_space + _backslash() + delete_space
        trail_backslash = delete_space + _backslash()
        lead_backslash = _backslash() + delete_space

        def _unix_slash():
            return pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश") + pynutil.insert("/")

        unix_seg_slash = delete_space + _unix_slash() + delete_space
        unix_lead_slash = _unix_slash() + delete_space
        unix_trail_slash = delete_space + _unix_slash()

        url_slash = (
            delete_space + pynutil.delete("फॉरवर्ड") + delete_space + pynutil.delete("स्लैश") + pynutil.insert("/")
        )

        lit_slash_seg = pynini.cross(" / ", "/")
        lit_hyphen_seg = pynini.cross(" - ", "-")

        dot = delete_space + (pynutil.delete("डॉट") | pynutil.delete("DOT")) + delete_space + pynutil.insert(".")
        dot_end_safe = (
            delete_space
            + (pynutil.delete("डॉट") | pynutil.delete("DOT"))
            + delete_zero_or_one_space
            + pynutil.insert(".")
        )
        hyphen = (
            delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-")
        )
        underscore = delete_space + pynutil.delete("अंडर") + delete_space + pynutil.delete("स्कोर") + pynutil.insert("_")
        at_sign = delete_space + pynutil.delete("एट") + delete_space
        x_sep = delete_space + pynutil.delete("एक्स") + pynutil.insert("x")
        literal_space = delete_space + pynutil.delete("स्पेस") + pynutil.insert(" ")
        open_bracket = (
            delete_space
            + pynutil.delete("ओपन")
            + delete_space
            + pynutil.delete("ब्रेकेट")
            + pynutil.insert("(")
            + delete_zero_or_one_space
        )
        close_bracket = (
            delete_space
            + pynutil.delete("क्लोज़")
            + delete_space
            + pynutil.delete("ब्रेकेट")
            + pynutil.insert(")")
            + delete_zero_or_one_space
        )
        dollar_sign = delete_space + pynutil.delete("डॉलर") + pynutil.insert("$")

        lit_open_paren = delete_space + pynutil.delete("(") + pynutil.insert("(") + delete_zero_or_one_space
        lit_close_paren = delete_space + pynutil.delete(")") + pynutil.insert(")")
        or_word = pynutil.delete("ओ") + delete_space + pynutil.delete("आर") + pynutil.insert("or")
        and_as_letters = pynutil.delete("एंड") + pynutil.insert("and")
        www_token = (
            pynutil.delete("डब्ल्यू")
            + delete_space
            + pynutil.delete("डब्ल्यू")
            + delete_space
            + pynutil.delete("डब्ल्यू")
            + pynutil.insert("www")
        )
        v_prefix = pynutil.delete("वी") + pynutil.insert("v")
        hp_token = pynutil.delete("एच") + delete_space + pynutil.delete("पी") + pynutil.insert("HP")
        tilde_delete = pynutil.delete("~") | pynutil.delete("टिल्ड")

        single_token = (
            pynutil.add_weight(server_map, 0.90)
            | pynutil.add_weight(common_map, 0.95)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        token_seq = single_token + pynini.closure(delete_space + single_token, 0)

        path_atom = (
            pynutil.add_weight(hp_token, 0.76)
            | pynutil.add_weight(www_token, 0.77)
            | pynutil.add_weight(or_word, 0.80)
            | pynutil.add_weight(and_as_letters, 0.84)
            | pynutil.add_weight(common_map, 0.90)
            | pynutil.add_weight(server_map, 0.92)
            | pynutil.add_weight(digit_words, 0.94)
            | pynutil.add_weight(digit_glyphs, 0.95)
            | pynutil.add_weight(latin_run, 0.97)
            | pynutil.add_weight(letter_map, 1.00)
        )
        path_atom_lower = (
            pynutil.add_weight(common_map_lower, 0.90)
            | pynutil.add_weight(server_map_lower, 0.92)
            | pynutil.add_weight(digit_words, 0.94)
            | pynutil.add_weight(digit_glyphs, 0.95)
            | pynutil.add_weight(latin_run_lower, 0.97)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        unix_path_atom = (
            pynutil.add_weight(www_token, 0.77)
            | pynutil.add_weight(or_word, 0.80)
            | pynutil.add_weight(and_as_letters, 0.84)
            | pynutil.add_weight(pynini.cross("CI", "c"), 0.86)
            | path_atom_lower
        )

        single_ext = (
            delete_space
            + pynutil.delete("डॉट")
            + pynutil.insert(".")
            + delete_space
            + path_atom_lower
            + pynini.closure(delete_space + path_atom_lower, 0)
        )
        ext_hyphen = (
            delete_space
            + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-")
            + delete_space
            + path_atom_lower
            + pynini.closure(delete_space + path_atom_lower, 0)
        )
        file_ext = single_ext + pynini.closure(single_ext | ext_hyphen, 0)

        win_hyphen = (
            delete_space
            + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-")
            + delete_space
            + path_atom
            + pynini.closure(delete_space + path_atom, 0)
        )
        win_underscore = (
            delete_space + pynutil.delete("अंडर") + delete_space + pynutil.delete("स्कोर") + pynutil.insert("_")
        )
        path_segment = (
            path_atom
            + pynini.closure(
                pynutil.add_weight(delete_space + path_atom, 1.0)
                | pynutil.add_weight(win_hyphen, 1.0)
                | pynutil.add_weight(win_underscore, 1.0)
                | pynutil.add_weight(literal_space, 1.0)
                | pynutil.add_weight(open_bracket, 1.0)
                | pynutil.add_weight(close_bracket, 1.0)
                | pynutil.add_weight(lit_open_paren, 1.0)
                | pynutil.add_weight(lit_close_paren, 1.0),
                0,
            )
            + pynini.closure(file_ext, 0, 1)
        )

        unix_hyphen = (
            delete_space
            + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-")
            + delete_space
            + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom, 0)
        )
        unix_underscore = (
            delete_space
            + pynutil.delete("अंडर")
            + delete_space
            + pynutil.delete("स्कोर")
            + pynutil.insert("_")
            + delete_space
            + unix_path_atom
            + pynini.closure(delete_space + unix_path_atom, 0)
        )
        version_seg = (
            v_prefix
            + unix_path_atom
            + pynini.closure(
                delete_space
                + pynutil.delete("डॉट")
                + pynutil.insert(".")
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
                pynutil.add_weight(delete_space + unix_path_atom, 1.0)
                | pynutil.add_weight(unix_hyphen, 1.0)
                | pynutil.add_weight(unix_underscore, 1.0),
                0,
            )
            + pynini.closure(file_ext, 0, 1)
        )

        windows_path_fst = (
            pynutil.insert("path: \"")
            + drive_letter
            + delete_space
            + pynutil.delete("कोलन")
            + pynutil.insert(":")
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
                pynutil.add_weight(delete_space + unix_path_atom, 1.0)
                | pynutil.add_weight(unix_hyphen, 1.0)
                | pynutil.add_weight(lit_hyphen_seg, 1.0),
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

        host_prefix_map = pynini.string_map(
            [
                ("एस आर वी", "srv"),
                ("डी बी", "db"),
                ("एल टी", "lt"),
                ("वेब", "web"),
                ("लैपटॉप", "laptop"),
                ("डेस्कटॉप", "desktop"),
                ("ई मेल", "email"),
            ]
        )

        domain_single = (
            pynutil.add_weight(server_map_lower, 0.90)
            | pynutil.add_weight(common_map_lower, 0.95)
            | pynutil.add_weight(letter_map_lower, 1.00)
        )
        domain_token_seq = domain_single + pynini.closure(delete_space + domain_single, 0)

        digit_then_letter = digit_seq + pynini.closure(delete_space + letter_map_lower, 0)

        first_label = (
            pynutil.add_weight(host_prefix_map, 0.80)
            | pynutil.add_weight(digit_seq + delete_space + letter_map_lower, 0.85)
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
            pynutil.add_weight(and_as_letters, 0.84)
            | pynutil.add_weight(digit_words, 0.88)
            | pynutil.add_weight(digit_glyphs, 0.88)
            | pynutil.add_weight(server_map, 0.90)
            | pynutil.add_weight(common_map, 0.95)
            | pynutil.add_weight(letter_map_lower, 0.84)
        )
        uname_sep = (
            (delete_space + pynutil.delete("डॉट") + delete_space + pynutil.insert("."))
            | (delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-"))
            | (delete_space + pynutil.delete("अंडर") + delete_space + pynutil.delete("स्कोर") + pynutil.insert("_"))
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
            pynutil.insert("ip: \"")
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
            pynutil.add_weight(pynutil.insert(".") + pynutil.delete("डॉट") + delete_space + token_seq, 0.90)
            | pynutil.add_weight(inline_domain_seg, 0.95)
            | pynutil.add_weight(path_segment_url, 1.00)
        )

        www_as_path_seg = www_token + dot + full_domain + pynini.closure(slash_with_word, 0)

        slash_with_word = url_slash + (
            pynutil.add_weight(delete_space + digit_seq + x_sep + delete_space + digit_seq, 0.20)
            | pynutil.add_weight(delete_space + digit_seq, 0.30)
            | pynutil.add_weight(
                delete_space + pynutil.insert(".") + pynutil.delete("डॉट") + delete_space + token_seq, 0.90
            )
            | pynutil.add_weight(delete_space + inline_domain_seg, 0.95)
            | pynutil.add_weight(delete_space + www_as_path_seg, 0.97)
            | pynutil.add_weight(delete_space + path_segment_url, 1.00)
        )

        hash_frag_body = token_seq + pynini.closure(hyphen + token_seq, 0)
        hash_frag = delete_space + pynutil.delete("हैशटैग") + pynutil.insert("#") + delete_space + hash_frag_body

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

        https_prefix = (
            pynutil.delete("एच")
            + delete_space
            + pynutil.delete("टी")
            + delete_space
            + pynutil.delete("टी")
            + delete_space
            + pynutil.delete("पी")
            + delete_space
            + pynutil.delete("एस")
            + delete_space
            + pynutil.delete("कोलन")
            + delete_space
            + pynutil.delete("फॉरवर्ड")
            + delete_space
            + pynutil.delete("स्लैश")
            + delete_space
            + pynutil.delete("फॉरवर्ड")
            + delete_space
            + pynutil.delete("स्लैश")
            + pynutil.insert("https://")
        )
        http_prefix = (
            pynutil.delete("एच")
            + delete_space
            + pynutil.delete("टी")
            + delete_space
            + pynutil.delete("टी")
            + delete_space
            + pynutil.delete("पी")
            + delete_space
            + pynutil.delete("कोलन")
            + delete_space
            + pynutil.delete("फॉरवर्ड")
            + delete_space
            + pynutil.delete("स्लैश")
            + delete_space
            + pynutil.delete("फॉरवर्ड")
            + delete_space
            + pynutil.delete("स्लैश")
            + pynutil.insert("http://")
        )
        protocol = pynutil.add_weight(https_prefix, 1.0) | pynutil.add_weight(http_prefix, 1.01)

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
            pynutil.add_weight(delete_space + chem_token, 1.0)
            | pynutil.add_weight(open_bracket, 1.0)
            | pynutil.add_weight(close_bracket, 1.0)
            | pynutil.add_weight(delete_space + pynutil.delete("इनदो") + pynutil.insert("("), 1.0)
            | pynutil.add_weight(delete_space + pynutil.delete("बाय") + pynutil.insert(")"), 1.0)
            | pynutil.add_weight(
                delete_space + (pynutil.delete("माइनस") | pynutil.delete("–")) + pynutil.insert("−"), 1.0
            ),
            0,
        )

        chem_spelled_fst = (
            pynutil.insert("domain: \"") + (chem_token + delete_space + chem_token + chem_more) + pynutil.insert("\"")
        )

        alnum_phrase_fst = pynutil.insert("domain: \"") + special_codes_map + pynutil.insert("\"")

        alnum_token = (
            pynutil.add_weight(digit_glyphs, 0.77)
            | pynutil.add_weight(digit_words, 0.10)
            | pynutil.add_weight(letter_map_upper, 0.84)
        )
        alnum_run = alnum_token + delete_space + alnum_token + pynini.closure(delete_space + alnum_token, 0)

        alnum_hyphen_ext = (
            delete_space
            + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन"))
            + pynutil.insert("-")
            + delete_space
            + alnum_token
            + pynini.closure(delete_space + alnum_token, 0)
        )
        alnum_body_start = alnum_run | (alnum_token + alnum_hyphen_ext)
        alnum_body = alnum_body_start + pynini.closure(
            pynutil.add_weight(alnum_hyphen_ext, 1.0)
            | pynutil.add_weight(
                delete_space
                + (pynutil.delete("डॉट") | pynutil.delete("DOT") | pynutil.delete("प्वाइंट"))
                + pynutil.insert(".")
                + delete_space
                + alnum_token
                + pynini.closure(delete_space + alnum_token, 0),
                1.0,
            )
            | pynutil.add_weight(
                delete_space
                + pynutil.delete("स्पेस")
                + pynutil.insert(" ")
                + delete_space
                + alnum_token
                + pynini.closure(delete_space + alnum_token, 0),
                1.0,
            )
            | pynutil.add_weight(lit_open_paren + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
            | pynutil.add_weight(lit_close_paren + alnum_token + pynini.closure(delete_space + alnum_token, 0), 1.0)
            | pynutil.add_weight(lit_close_paren, 1.0),
            0,
        )
        alnum_letterdigit_fst = pynutil.insert("domain: \"") + alnum_body + pynutil.insert("\"")

        chem_fst = (
            pynutil.add_weight(pynutil.insert("domain: \"") + chem_named_map + pynutil.insert("\""), 0.04)
            if chem_named_map is not None
            else pynini.accep("")
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
            | pynutil.add_weight(alnum_phrase_fst, 0.05)
            | pynutil.add_weight(chem_spelled_fst, 1.18)
            | pynutil.add_weight(alnum_letterdigit_fst, 0.90)
            | pynutil.add_weight(plain_fst, 1.30)
            | chem_fst
        )

        self.fst = self.add_tokens(graph).optimize()
