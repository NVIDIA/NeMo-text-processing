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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, delete_space
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_TA_LETTER, PLUS_WORD, TO_ASCII_DIGITS
from nemo_text_processing.text_normalization.ta.utils import get_abs_path as tn_abs_path

# Spoken zero variants beyond the telephone table's word.
ZERO_WORDS = ("பூஜ்ஜியம்",)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying spoken digit strings, e.g.
        ஒன்பது ஒன்பது நான்கு ... பூஜ்யம் -> telephone { number_part: "9943206870" }
        பிளஸ் தொண்ணூற்றொன்று ஒன்பது எட்டு ... -> telephone { country_code: "+91" number_part: "9876543210" }
        பூஜ்யம் பூஜ்யம் ஏழு -> telephone { number_part: "007" }

    Three or more digit words in a row are a digit string (phone, PIN, OTP, 007). Reads the
    TN ``data/telephone/number.tsv`` from the spoken side.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="telephone", kind="classify")

        digit_words = pynini.invert(pynini.string_file(tn_abs_path("data/telephone/number.tsv"))).optimize()
        digit = digit_words @ TO_ASCII_DIGITS
        for word in ZERO_WORDS:
            digit |= pynini.cross(word, "0")
        digit = digit.optimize()

        # A case suffix on the last digit word is carried over (...பூஜ்யத்தில் -> ...0ல்), through
        # the cardinal's own suffix reading.
        suffixed_digit = cardinal.words_to_digits_suffixed @ (NEMO_DIGIT + pynini.closure(NEMO_TA_LETTER, 1))
        last = pynini.union(digit, suffixed_digit)
        # After a country code the number is a 10-digit mobile or 11-digit landline.
        number = digit + pynini.closure(delete_space + digit, 1) + delete_space + last
        cc_number = digit + pynini.closure(delete_space + digit, 8, 9) + delete_space + last

        plus = pynini.cross(PLUS_WORD, "+")
        # The plus word followed by one to three digit words, or by a spoken number.
        code_digits = pynini.union(
            digit + pynini.closure(delete_space + digit, 0, 2),
            cardinal.words_to_digits @ pynini.closure(NEMO_DIGIT, 1, 3),
        )
        country_code = pynutil.insert("country_code: \"") + plus + delete_space + code_digits + pynutil.insert("\"")

        number_part = pynutil.insert("number_part: \"") + number + pynutil.insert("\"")
        cc_number_part = pynutil.insert("number_part: \"") + cc_number + pynutil.insert("\"")
        graph = number_part | (country_code + pynutil.insert(" ") + delete_space + cc_number_part)
        # A standalone two- or three-digit country code: பிளஸ் தொண்ணூற்றொன்று -> +91.
        standalone = pynutil.insert("country_code: \"") + plus + delete_space
        standalone += pynini.union(
            digit + pynini.closure(delete_space + digit, 1, 2),
            cardinal.words_to_digits @ pynini.closure(NEMO_DIGIT, 2, 3),
        ) + pynutil.insert("\"")
        graph |= pynutil.add_weight(standalone, 0.2)
        self.fst = self.add_tokens(graph).optimize()
