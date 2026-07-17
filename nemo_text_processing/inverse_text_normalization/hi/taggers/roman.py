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
    GraphFst,
    delete_space,
    insert_space,
    integer_to_devanagari,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, load_labels


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying spoken numbers as Roman numerals
    when they follow a small, fixed set of context key words (chapter, volume,
    class numbering). The conversion is deliberately restricted to these
    predictable contexts; regnal, papal and product names (e.g. भास्कर-II) are a
    documented limitation because the same number is ambiguous between Arabic and
    Roman form. Numbers above MAX_NUMBER have no Roman form, so the whole number
    falls back to Arabic (Devanagari) digits instead of leaving an in-range prefix
    dangling.
        e.g. अध्याय तीन -> tokens { roman { key_cardinal: "अध्याय" integer: "III" } }
        e.g. कक्षा दस -> tokens { roman { key_cardinal: "कक्षा" integer: "X" } }
        e.g. अध्याय चार हजार -> tokens { roman { key_cardinal: "अध्याय" integer: "४०००" } }

    Args:
        cardinal: CardinalFst, used to read spoken numbers.
    """

    MAX_NUMBER = 3999

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="roman", kind="classify")

        key_words = [label[0] for label in load_labels(get_abs_path("data/roman/key_words.tsv"))]
        key_words_fst = pynini.union(*[pynini.accep(word) for word in key_words]).optimize()

        value_to_roman = {
            int(value): roman for roman, value in load_labels(get_abs_path("data/roman/roman_numerals.tsv"))
        }

        devanagari_to_roman = pynini.string_map(
            [
                (integer_to_devanagari(value), self._int_to_roman(value, value_to_roman))
                for value in range(1, self.MAX_NUMBER + 1)
            ]
        ).optimize()

        number_to_devanagari = cardinal.graph_no_exception
        in_range_to_roman = pynini.compose(number_to_devanagari, devanagari_to_roman).optimize()

        roman_range_devanagari = pynini.determinize(
            pynini.project(devanagari_to_roman, "input").rmepsilon()
        ).optimize()
        all_devanagari = pynini.project(number_to_devanagari, "output").optimize()
        above_range_devanagari = pynini.difference(all_devanagari, roman_range_devanagari).optimize()
        above_range_to_arabic = pynini.compose(number_to_devanagari, above_range_devanagari).optimize()

        spoken_to_roman = pynini.union(in_range_to_roman, above_range_to_arabic).optimize()

        graph = (
            pynutil.insert('key_cardinal: "')
            + key_words_fst
            + pynutil.insert('"')
            + delete_space
            + insert_space
            + pynutil.insert('integer: "')
            + spoken_to_roman
            + pynutil.insert('"')
        )
        self.fst = self.add_tokens(graph).optimize()

    def _int_to_roman(self, number, value_to_roman):
        roman = ""
        for value in sorted(value_to_roman, reverse=True):
            while number >= value:
                roman += value_to_roman[value]
                number -= value
        return roman
