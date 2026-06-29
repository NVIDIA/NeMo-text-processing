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
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, load_labels


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying spoken numbers as Roman numerals
    when they follow a small, fixed set of context key words (chapter, volume,
    class numbering). The conversion is deliberately restricted to these
    predictable contexts; regnal, papal and product names (e.g. भास्कर-II) are a
    documented limitation because the same number is ambiguous between Arabic and
    Roman form.
        e.g. अध्याय तीन -> tokens { roman { key: "अध्याय" integer: "III" } }
        e.g. कक्षा दस -> tokens { roman { key: "कक्षा" integer: "X" } }

    Args:
        cardinal: CardinalFst, used to read spoken numbers.
    """

    MAX_NUMBER = 3999

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="roman", kind="classify")

        key_words = [label[0] for label in load_labels(get_abs_path("data/roman/key_words.tsv"))]
        key_words_fst = pynini.union(*[pynini.accep(word) for word in key_words]).optimize()

        roman_to_value = {
            roman: int(value) for roman, value in load_labels(get_abs_path("data/roman/roman_numerals.tsv"))
        }
        value_to_roman = {value: roman for roman, value in roman_to_value.items()}

        not_quote = pynini.closure(pynini.difference(NEMO_SIGMA, pynini.accep('"')), 1)
        strip_cardinal_tags = pynutil.delete('cardinal { integer: "') + not_quote + pynutil.delete('" }')
        cardinal_to_devanagari = pynini.compose(cardinal.fst, strip_cardinal_tags).optimize()

        single_digit_to_devanagari = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        )
        glyph_to_ascii = pynini.union(
            *[pynini.cross(glyph, str(value)) for value, glyph in enumerate(DEVANAGARI_DIGIT)]
        )
        devanagari_to_ascii = pynini.cdrewrite(glyph_to_ascii, "", "", NEMO_SIGMA)
        spoken_to_ascii = pynini.compose(
            cardinal_to_devanagari | single_digit_to_devanagari, devanagari_to_ascii
        ).optimize()

        ascii_to_roman = pynini.string_map(
            [(str(value), self._int_to_roman(value, value_to_roman)) for value in range(1, self.MAX_NUMBER + 1)]
        ).optimize()
        spoken_to_roman = pynini.compose(spoken_to_ascii, ascii_to_roman).optimize()

        graph = (
            pynutil.insert("key: \"")
            + key_words_fst
            + pynutil.insert("\"")
            + delete_space
            + insert_space
            + pynutil.insert("integer: \"")
            + spoken_to_roman
            + pynutil.insert("\"")
        )
        self.fst = self.add_tokens(graph).optimize()

    def _int_to_roman(self, number, value_to_roman):
        roman = ""
        for value in sorted(value_to_roman, reverse=True):
            while number >= value:
                roman += value_to_roman[value]
                number -= value
        return roman
