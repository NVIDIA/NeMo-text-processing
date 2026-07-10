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
    DEVANAGARI_LETTER,
    DIGIT_GLYPH_TO_ASCII,
    DIGIT_WORD_TO_DEVANAGARI,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    load_symbols,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import TO_UPPER


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial strings, whose segments are
    joined by a hyphen (a literal "-" or the spoken word "हाइफ़न").
        e.g. कोविड-उन्नीस -> tokens { name: "कोविड-19" }
        e.g. ब्रह्मोस हाइफ़न १ -> tokens { name: "ब्रह्मोस-1" }
        e.g. एक-आठ सौ-पाँच सौ पचपन -> tokens { name: "1-800-555" }
        e.g. दो स्क्वेर्ड -> tokens { name: "2^2" }
        e.g. आई ए तीन दो -> tokens { name: "IA32" }
        e.g. जी एस ए टी हाइफ़न एक आठ -> tokens { name: "GSAT-18" }

    Args:
        cardinal: CardinalFst, used to read spoken numbers.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="serial", kind="classify")

        cardinal_to_devanagari = cardinal.graph.optimize()

        devanagari_to_ascii = pynini.cdrewrite(DIGIT_GLYPH_TO_ASCII, "", "", NEMO_SIGMA)
        spoken_number = pynini.compose(cardinal_to_devanagari | DIGIT_WORD_TO_DEVANAGARI, devanagari_to_ascii)
        devanagari_numeral = pynini.closure(DIGIT_GLYPH_TO_ASCII, 1)
        number = (spoken_number | devanagari_numeral).optimize()
        number_words = pynini.arcmap(pynini.project(number, "input"), map_type="rmweight").optimize()

        devanagari_word = pynini.closure(DEVANAGARI_LETTER, 1)

        letter_names = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/letters.tsv")), "output"
        ).optimize()
        word = pynini.difference(devanagari_word, (number_words | letter_names).optimize()).optimize()

        segment = word | number

        sym = load_symbols(get_abs_path("data/electronic/symbols.tsv"))
        word_hyphen = delete_space + sym["hyphen"] + delete_space
        delimiter = pynini.accep("-") | word_hyphen
        serial_core = segment + pynini.closure(delimiter + segment, 1)

        power_special = pynini.string_file(get_abs_path("data/serial/power_special.tsv"))
        power_prefix = pynini.string_file(get_abs_path("data/serial/power.tsv"))
        power_generic = power_prefix + delete_space + number
        power_suffix = delete_space + (power_special | power_generic)
        power_graph = number + power_suffix

        digit_words = (DIGIT_WORD_TO_DEVANAGARI @ DIGIT_GLYPH_TO_ASCII).optimize()
        letter_map_upper = (
            pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert() @ TO_UPPER
        ).optimize()

        alnum_token = DIGIT_GLYPH_TO_ASCII | digit_words | letter_map_upper
        alnum_run = alnum_token + pynini.closure(delete_space + alnum_token, 1)

        alnum_hyphen_ext = (
            delete_space + sym["hyphen"] + delete_space + alnum_token + pynini.closure(delete_space + alnum_token)
        )
        alnum_point_ext = (
            delete_space + sym["point"] + delete_space + alnum_token + pynini.closure(delete_space + alnum_token)
        )
        alnum_body = (alnum_run | (alnum_token + alnum_hyphen_ext)) + pynini.closure(
            alnum_hyphen_ext | alnum_point_ext
        )

        # restrict to serials mixing letters and digits, so digit-only and
        # letter-only inputs stay with the cardinal/telephone/word classes
        ascii_alpha = pynini.union(
            *[chr(c) for c in range(ord("A"), ord("Z") + 1)],
            *[chr(c) for c in range(ord("a"), ord("z") + 1)],
        )
        ascii_digit = pynini.union(*[str(d) for d in range(10)])
        contains_alpha = NEMO_SIGMA + ascii_alpha + NEMO_SIGMA
        contains_digit = NEMO_SIGMA + ascii_digit + NEMO_SIGMA
        alnum_mix = pynini.intersect(contains_alpha, contains_digit).optimize()
        alnum_body = (alnum_body @ alnum_mix).optimize()

        graph = pynutil.insert("name: \"") + (serial_core | power_graph | alnum_body) + pynutil.insert("\"")
        self.fst = graph.optimize()
