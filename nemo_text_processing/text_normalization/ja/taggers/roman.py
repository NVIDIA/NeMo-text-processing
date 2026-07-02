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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.ja.utils import get_abs_path, load_labels


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying Roman numerals in supported contexts.

    Examples:
        第III章 -> name: "第三章"
        Chapter IV -> name: "Chapter 四"
        Henry VIII -> name: "Henry 八世"
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        roman_values = {
            roman: int(value) for roman, value in load_labels(get_abs_path("data/roman/roman_numerals.tsv"))
        }
        valid_roman_pairs = []
        for number in range(1, 4000):
            roman = self._int_to_roman(number, roman_values)
            valid_roman_pairs.append((roman, str(number)))
            valid_roman_pairs.append((roman.lower(), str(number)))

        roman_to_number = pynini.string_map(valid_roman_pairs).optimize()
        roman_to_cardinal = roman_to_number @ cardinal.just_cardinals

        japanese_suffix = pynini.union("章", "条", "巻", "回")
        japanese_context = pynini.accep("第") + roman_to_cardinal + japanese_suffix

        key_cardinal = pynini.union(
            *[pynini.accep(x[0]) for x in load_labels(get_abs_path("data/roman/key_cardinal.tsv"))]
        )
        key_ordinal = pynini.union(
            *[pynini.accep(x[0]) for x in load_labels(get_abs_path("data/roman/key_ordinal.tsv"))]
        )

        cardinal_context = key_cardinal + pynutil.delete(" ") + insert_space + roman_to_cardinal
        ordinal_context = key_ordinal + pynutil.delete(" ") + insert_space + roman_to_cardinal + pynutil.insert("世")
        preserve = pynini.string_file(get_abs_path("data/roman/preserve.tsv"))

        graph = japanese_context | cardinal_context | ordinal_context | preserve
        self.fst = (pynutil.insert('name: "') + graph.optimize() + pynutil.insert('"')).optimize()

    @staticmethod
    def _int_to_roman(number: int, roman_values: dict) -> str:
        value_to_roman = sorted(((value, roman) for roman, value in roman_values.items()), reverse=True)
        result = []
        remaining = number
        for value, roman in value_to_roman:
            while remaining >= value:
                result.append(roman)
                remaining -= value
        return "".join(result)
