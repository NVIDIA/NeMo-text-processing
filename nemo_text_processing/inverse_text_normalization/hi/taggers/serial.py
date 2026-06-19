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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial strings, whose segments are
    joined by a hyphen (a literal "-" or the spoken word "हाइफ़न").
        e.g. कोविड-उन्नीस -> tokens { serial { name: "कोविड-19" } }
        e.g. ब्रह्मोस हाइफ़न १ -> tokens { serial { name: "ब्रह्मोस-1" } }
        e.g. एक-आठ सौ-पाँच सौ पचपन -> tokens { serial { name: "1-800-555" } }
        e.g. दो स्क्वेर्ड -> tokens { serial { name: "2^2" } }

    Args:
        cardinal: CardinalFst, used to read spoken numbers.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="serial", kind="classify")

        not_quote = pynini.closure(pynini.difference(NEMO_SIGMA, pynini.accep('"')), 1)
        strip_cardinal_tags = pynutil.delete('cardinal { integer: "') + not_quote + pynutil.delete('" }')
        cardinal_to_devanagari = pynini.compose(cardinal.fst, strip_cardinal_tags).optimize()

        single_digit_to_devanagari = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        )
        glyph_to_ascii = pynini.union(*[pynini.cross(glyph, str(value)) for value, glyph in enumerate(DEVANAGARI_DIGIT)])
        devanagari_to_ascii = pynini.cdrewrite(glyph_to_ascii, "", "", NEMO_SIGMA)
        spoken_number = pynini.compose(cardinal_to_devanagari | single_digit_to_devanagari, devanagari_to_ascii)
        devanagari_numeral = pynini.closure(glyph_to_ascii, 1)
        number = (spoken_number | devanagari_numeral).optimize()
        number_words = pynini.arcmap(pynini.project(number, "input"), map_type="rmweight").optimize()


        devanagari_letter = pynini.union(
            *[chr(c) for c in range(0x0900, 0x0966)],
            *[chr(c) for c in range(0x0970, 0x0980)],
        )
        devanagari_word = pynini.closure(devanagari_letter, 1)

        letter_names = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/letters.tsv")), "output"
        ).optimize()
        word = pynini.difference(devanagari_word, (number_words | letter_names).optimize()).optimize()

        segment = word | number


        word_hyphen = delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-")
        delimiter = pynini.accep("-") | word_hyphen
        serial_core = segment + pynini.closure(delimiter + segment, 1)

        power_special = pynini.string_file(get_abs_path("data/serial/power_special.tsv"))
        power_generic = pynutil.delete("टु द पावर") + delete_space + pynutil.insert("^") + number
        power_suffix = delete_space + (power_special | power_generic)
        power_graph = number + power_suffix

        graph = pynutil.insert("name: \"") + (serial_core | power_graph) + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()
