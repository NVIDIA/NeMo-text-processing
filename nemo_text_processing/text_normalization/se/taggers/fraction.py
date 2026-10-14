# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Jim O'Regan for Språkbanken Tal
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

import unicodedata

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst, convert_space
from nemo_text_processing.text_normalization.se.utils import get_abs_path


def fraction_nfkc_graph():
    symbols = "½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒"
    return pynini.string_map((symbol, unicodedata.normalize("NFKC", symbol)) for symbol in symbols)


def ascii_fraction_slash(graph):
    slash_to_ascii = pynini.cdrewrite(pynini.cross("\u2044", "/"), "", "", NEMO_SIGMA)
    return graph @ slash_to_ascii


class FractionFst(GraphFst):
    """Classifies vulgar fractions and mixed-number expressions."""

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="name", kind="classify", deterministic=deterministic)

        denominator_input = pynini.project(ordinal.graph_bare_ordinals, "input")
        denominator_stem = ordinal.graph_bare_ordinals @ pynini.cdrewrite(
            pynini.cross("t", "das"), "", "[EOS]", NEMO_SIGMA
        )
        denominator_nominative = pynini.cross("2", "bealli") | (
            pynini.difference(denominator_input, "2") @ denominator_stem
        )
        denominator_genitive = pynini.cross("2", "beali") | (
            pynini.difference(denominator_input, "2")
            @ denominator_stem
            @ pynini.cdrewrite(pynutil.insert("a"), "", "[EOS]", NEMO_SIGMA)
        )

        one_fraction = pynutil.delete("1/") + denominator_nominative
        other_fraction = cardinal.graph + pynutil.delete("/") + pynutil.insert(" ") + denominator_genitive
        spoken_fraction = (one_fraction | other_fraction).optimize()
        nfkc_graph = fraction_nfkc_graph()
        ascii_graph = ascii_fraction_slash(nfkc_graph)
        symbol_fraction = (ascii_graph @ spoken_fraction).optimize()
        ascii_forms = pynini.project(ascii_graph, "output")
        unicode_forms = pynini.project(nfkc_graph, "output")
        raw_fraction = (ascii_forms | ascii_fraction_slash(unicode_forms)) @ spoken_fraction

        integer_input = pynini.project(cardinal.graph, "input")
        generic_integer = pynini.difference(integer_input, pynini.union("1", "2")) @ cardinal.graph
        mixed = generic_integer + pynutil.delete(" ") + pynutil.insert(" ja ") + symbol_fraction

        graph = symbol_fraction | raw_fraction | mixed | pynini.string_file(get_abs_path("data/numbers/fraction.tsv"))
        if not deterministic:
            graph |= pynini.string_file(get_abs_path("data/numbers/fraction_nd.tsv"))
            graph |= cardinal.graph + pynutil.delete(" ") + pynutil.insert(" ja ") + symbol_fraction

        self.graph = graph.optimize()
        self.fst = pynutil.insert('name: "') + convert_space(self.graph) + pynutil.insert('"')
