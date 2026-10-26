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
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    load_symbols,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, NEMO_DIGIT, TO_UPPER


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
        e.g. रेडमी के तीन -> tokens { name: "रेडमी K3" }
        e.g. चार के वीडियो -> tokens { name: "4K वीडियो" }

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

        letters = pynini.string_file(get_abs_path("data/electronic/letters.tsv"))
        letter_names = pynini.project(letters, "output").optimize()
        word = pynini.difference(devanagari_word, (number_words | letter_names).optimize()).optimize()

        segment = word | number

        sym = load_symbols(get_abs_path("data/electronic/symbols.tsv"))
        word_hyphen = delete_space + sym["hyphen"] + delete_space
        delimiter = sym["lithyphen"] | word_hyphen
        serial_core = segment + pynini.closure(delimiter + segment, 1)

        power_special = pynini.string_file(get_abs_path("data/serial/power_special.tsv"))
        power_prefix = pynini.string_file(get_abs_path("data/serial/power.tsv"))
        power_generic = power_prefix + delete_space + number
        power_suffix = delete_space + (power_special | power_generic)
        power_graph = number + power_suffix

        digit_words = (DIGIT_WORD_TO_DEVANAGARI @ DIGIT_GLYPH_TO_ASCII).optimize()
        letter_map_upper = (letters.invert() @ TO_UPPER).optimize()

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

        contains_alpha = NEMO_SIGMA + NEMO_ALPHA + NEMO_SIGMA
        contains_digit = NEMO_SIGMA + NEMO_DIGIT + NEMO_SIGMA
        alnum_mix = pynini.intersect(contains_alpha, contains_digit).optimize()
        lone_k = (pynini.closure(NEMO_DIGIT) + pynini.accep("K") + pynini.closure(NEMO_DIGIT)).optimize()
        alnum_mix = pynini.difference(alnum_mix, lone_k).optimize()
        alnum_body = (alnum_body @ alnum_mix).optimize()

        k_letter = delete_space + pynini.cross("के", "K") + delete_space
        nbsp = pynutil.insert(NEMO_NON_BREAKING_SPACE)
        k_brands = pynini.string_file(get_abs_path("data/serial/k_brands.tsv"))
        brand_model = k_brands + nbsp + k_letter + number
        k_screen_words = pynini.string_file(get_abs_path("data/serial/k_screen_words.tsv"))
        resolution = (number @ pynini.union("4", "8")) + k_letter + nbsp + k_screen_words
        k_context = (brand_model | resolution).optimize()

        graph = (
            pynutil.insert("name: \"") + (serial_core | power_graph | alnum_body | k_context) + pynutil.insert("\"")
        )
        self.fst = graph.optimize()
