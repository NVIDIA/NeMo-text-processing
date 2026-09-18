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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_ALL_DIGIT, NEMO_ALL_ZERO, PLUS_WORD
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        9943206870 -> telephone { number_part: "ஒன்பது ஒன்பது நான்கு மூன்று இரண்டு பூஜ்யம் ஆறு எட்டு ஏழு பூஜ்யம்" }
        +91 9876543210 -> telephone { country_code: "பிளஸ் ஒன்பது ஒன்று" number_part: "..." }
        044-28230000 -> telephone { number_part: "பூஜ்யம் நான்கு நான்கு இரண்டு எட்டு ..." }
        +91 -> telephone { country_code: "பிளஸ் தொண்ணூற்றொன்று" }

    Reads ``data/telephone/number.tsv`` (digit in either script -> word). Indian mobile,
    landline and toll-free shapes are read digit by digit; a case suffix on the number lands on
    the last digit word. A local number written without its STD code is read this way only in
    the 3-4 shape (123-4567), so a 4-4 span such as the year range 2010-2020 stays a range.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        single_digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv")).optimize()
        mobile_first_digit = pynini.union(*"6789", *"௬௭௮௯")
        zero_digit = NEMO_ALL_ZERO
        one_word = pynini.union("1", "௧") @ single_digit_to_word

        digit_word = single_digit_to_word + insert_space
        last_digit_word = single_digit_to_word
        delete_sep = pynutil.delete(pynini.union("-", " "))
        optional_sep = pynini.closure(delete_sep, 0, 1)

        # A case suffix on the number lands on the last digit word (9876543210க்கு -> ...பூஜ்யத்துக்கு).
        last_digit_suffixed = cardinal.attach_case_suffix(last_digit_word)

        def shapes(last: 'pynini.FstLike'):
            # 10-digit mobile starting 6-9, optionally after a trunk 0 (09876543210); a 5-5 split
            # with space or dash is common.
            trunk = pynini.closure((zero_digit @ single_digit_to_word) + insert_space, 0, 1)
            mobile = (
                trunk
                + (mobile_first_digit @ single_digit_to_word)
                + insert_space
                + pynini.closure(digit_word, 3, 3)
                + digit_word
                + optional_sep
                + pynini.closure(digit_word, 4, 4)
                + last
            )

            # Landline: STD code starting 0 (2-4 digits, optionally in parentheses), a dash or
            # space, then a 6-8 digit subscriber number optionally split once.
            std_digits = (zero_digit @ single_digit_to_word) + insert_space + pynini.closure(digit_word, 1, 3)
            std_code = std_digits | (pynutil.delete("(") + std_digits + pynutil.delete(")"))
            # A hyphen split inside the subscriber is only the 4-4 shape (2823-0000), so a date
            # like 01-04-2024 never reads as a landline.
            subscriber = (
                pynini.closure(digit_word, 2, 4)
                + pynini.closure(pynutil.delete(" "), 0, 1)
                + pynini.closure(digit_word, 2, 3)
                + last
            )
            subscriber |= (
                pynini.closure(digit_word, 4, 4) + pynutil.delete("-") + pynini.closure(digit_word, 3, 3) + last
            )
            landline = std_code + optional_sep + subscriber

            # A local number written without its STD code: three digits, a dash, then four
            # (123-4567). Only the 3-4 split is read this way, so a 4-4 span such as the year
            # range 2010-2020 is left to the range class.
            landline |= (
                pynini.closure(digit_word, 3, 3) + pynutil.delete("-") + pynini.closure(digit_word, 3, 3) + last
            )

            # Toll-free: 1800-XXX-XXXX / 1-800-XXX-XXXX.
            toll_free = (
                one_word
                + insert_space
                + optional_sep
                + pynini.closure(digit_word, 3, 3)
                + delete_sep
                + pynini.closure(digit_word, 3, 3)
                + delete_sep
                + pynini.closure(digit_word, 3, 3)
                + last
            )
            # Toll-free 1800-11-4000 / 1800 11 4000: 1800 + 2-3 digits + 3-4 digits.
            toll_free |= (
                one_word
                + insert_space
                + pynini.closure(digit_word, 3, 3)
                + delete_sep
                + pynini.closure(digit_word, 2, 3)
                + delete_sep
                + pynini.closure(digit_word, 2, 3)
                + last
            )

            # After a country code the STD code drops its leading zero: +91-44-28230000,
            # optionally in parentheses: +91 (44) 2823 0000.
            std_digits_no_zero = pynini.closure(digit_word, 2, 4)
            std_no_zero = (
                (std_digits_no_zero | pynutil.delete("(") + std_digits_no_zero + pynutil.delete(")"))
                + delete_sep
                + subscriber
            )
            return pynini.union(mobile, landline, toll_free), std_no_zero

        plain, cc_landline = shapes(last_digit_word)
        suffixed, cc_landline_suffixed = shapes(last_digit_suffixed)

        country_code = (
            pynutil.insert("country_code: \"")
            + pynini.cross("+", PLUS_WORD)
            + insert_space
            + pynini.closure(digit_word, 0, 2)
            + last_digit_word
            + pynutil.insert("\" ")
            + pynini.closure(delete_space | pynutil.delete("-"), 0, 1)
        )

        def number_part(inner: 'pynini.FstLike') -> 'pynini.FstLike':
            return pynutil.insert("number_part: \"") + inner + pynutil.insert("\"")

        graph = pynini.union(
            pynutil.add_weight(country_code + number_part(plain | cc_landline), 0.1),
            pynutil.add_weight(number_part(plain), 0.1),
            pynutil.add_weight(country_code + number_part(suffixed | cc_landline_suffixed), 0.2),
            pynutil.add_weight(number_part(suffixed), 0.2),
        )

        # A + before 11-13 glued digits with no separator (+919876543210) reads digit by digit;
        # a shorter run (+91, +5) is a signed cardinal.
        standalone_cc = (
            pynutil.insert("country_code: \"")
            + pynini.cross("+", PLUS_WORD)
            + insert_space
            + pynini.compose(pynini.closure(NEMO_ALL_DIGIT, 11, 13), cardinal.digit_by_digit)
            + pynutil.insert("\"")
        )
        graph |= pynutil.add_weight(standalone_cc, 0.3)

        self.final = graph.optimize()
        self.fst = self.add_tokens(self.final)
