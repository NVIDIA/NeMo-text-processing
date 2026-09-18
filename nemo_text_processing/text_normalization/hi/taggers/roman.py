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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, convert_space, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path, load_labels


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying Roman numerals in Hindi text.
        e.g. भास्कर-II    -> tokens { roman { key_cardinal: "भास्कर" integer: "II" } }
        e.g. कक्षा XII    -> tokens { roman { key_cardinal: "कक्षा" integer: "XII" } }
        e.g. XIIवीं कक्षा -> tokens { roman { integer: "XII" default_ordinal: "बारहवीं" key_cardinal: "कक्षा" } }
        e.g. IVथी कक्षा   -> tokens { roman { integer: "IV" default_ordinal: "चौथी" key_cardinal: "कक्षा" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        components = load_labels(get_abs_path("data/roman/roman_components.tsv"))
        ones = pynini.string_map(components[0:9]).optimize()
        tens = pynini.string_map(components[9:18]).optimize()
        hundreds = pynini.string_map(components[18:27]).optimize()
        thousands = pynini.string_map(components[27:30]).optimize()

        zero = pynutil.insert("0")
        opt_hundreds = hundreds | zero
        opt_tens = tens | zero
        opt_ones = ones | zero

        roman_to_arabic = pynini.union(
            thousands + opt_hundreds + opt_tens + opt_ones, hundreds + opt_tens + opt_ones, tens + opt_ones, ones
        ).optimize()

        roman_to_spoken_fst = pynini.compose(roman_to_arabic, cardinal.graph_without_leading_zeros).optimize()

        devanagari_chars = pynini.project(
            pynini.string_file(get_abs_path("data/serial/chars.tsv")), "input"
        ).optimize()

        devanagari_word = pynini.closure(devanagari_chars, 1).optimize()

        devanagari_phrase = (
            devanagari_word + pynini.closure((pynini.accep(" ") | pynini.accep("-")) + devanagari_word)
        ).optimize()

        separator = (pynini.accep("-") | pynini.accep(" ")).optimize()

        key_before_numeral = (
            pynutil.insert("preserve_order: true ")
            + pynutil.insert('key_cardinal: "')
            + convert_space(devanagari_phrase)
            + pynutil.insert('"')
            + pynutil.delete(separator)
            + insert_space
            + pynutil.insert('integer: "')
            + roman_to_spoken_fst
            + pynutil.insert('"')
        ).optimize()

        numeral_before_key = (
            pynutil.insert("preserve_order: true ")
            + pynutil.insert('integer: "')
            + roman_to_spoken_fst
            + pynutil.insert('"')
            + pynutil.delete(separator)
            + insert_space
            + pynutil.insert('key_cardinal: "')
            + convert_space(devanagari_phrase)
            + pynutil.insert('"')
        ).optimize()

        exception_rows = load_labels(get_abs_path("data/roman/roman_ordinal_exceptions.tsv"))

        exception_graphs = []
        for fused, spoken_word in exception_rows:
            exception_graphs.append(
                pynutil.insert('integer: "-"')
                + insert_space
                + pynutil.insert('default_ordinal: "' + spoken_word + '"')
                + pynutil.delete(fused)
            )
        glued_ordinal_exceptions_graph = pynini.union(*exception_graphs).optimize()

        suffixes_fst = pynini.union(
            pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv")),
            pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv")),
        ).optimize()

        glued_ordinal_regular_graph = (
            pynutil.insert('integer: "-"')
            + insert_space
            + pynutil.insert('default_ordinal: "')
            + (roman_to_spoken_fst + suffixes_fst)
            + pynutil.insert('"')
        ).optimize()

        roman_glued_ordinal_fields = pynini.union(
            pynutil.add_weight(glued_ordinal_exceptions_graph, -0.1),
            glued_ordinal_regular_graph,
        ).optimize()

        roman_glued_ordinal = (
            pynutil.insert("preserve_order: true ")
            + roman_glued_ordinal_fields
            + pynini.closure(
                pynutil.delete(" ")
                + insert_space
                + pynutil.insert('key_cardinal: "')
                + convert_space(devanagari_phrase)
                + pynutil.insert('"'),
                0,
                1,
            )
        ).optimize()

        graph = pynini.union(key_before_numeral, numeral_before_key, roman_glued_ordinal).optimize()

        self.fst = self.add_tokens(graph).optimize()
