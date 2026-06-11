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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial strings in Hindi.
    Handles Devanagari-numeric mixtures, complex delimited number chains,
    symbols, and powers. Supports both ASCII (0-9) and Devanagari (०-९) digits.

        e.g. कोविड-19  -> tokens { name: "कोविड-उन्नीस" }
        e.g. 5जी       -> tokens { name: "पाँच जी" }
        e.g. ३जी       -> tokens { name: "तीन जी" }
        e.g. 2^2       -> tokens { name: "दो स्क्वेर्ड" }
        e.g. 2^4       -> tokens { name: "दो टु द पावर चार" }
        e.g. 1-800-555 -> tokens { name: "एक-आठ सौ-पाँच सौ पचपन" }

    Note: Pure Latin-alpha + digit patterns (A12, B-60) are intentionally
    excluded here so they fall through to the electronic classifier.
    """

    def __init__(
        self,
        cardinal: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        digit_graph = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero_graph = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        devanagari_digits = pynini.project(
            pynini.union(digit_graph, zero_graph),
            "input",
        ).optimize()

        any_digit = pynini.union(NEMO_DIGIT, devanagari_digits).optimize()

        not_quote = pynini.closure(pynini.difference(NEMO_SIGMA, pynini.accep('"')), 1)
        strip_cardinal_tags = pynutil.delete('cardinal { integer: "') + not_quote + pynutil.delete('" }')

        pure_cardinal_words = pynini.compose(cardinal.fst, strip_cardinal_tags).optimize()

        length_filter = pynini.closure(any_digit, 1, 3)
        limited_cardinal = pynini.compose(length_filter, pure_cardinal_words).optimize()

        num_graph = limited_cardinal

        symbols_graph = pynini.string_file(get_abs_path("data/serial/special_symbols.tsv")).optimize()

        devanagari_chars = pynini.project(
            pynini.string_file(get_abs_path("data/serial/chars.tsv")),
            "input",
        ).optimize()

        letter_graph = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        latin_letters = letter_graph + pynini.closure(pynutil.insert(" ") + letter_graph)
        latin_letters = latin_letters.optimize()

        devanagari_word = pynini.closure(devanagari_chars, 2).optimize()

        delimiter = (pynini.accep("-") | pynini.accep("/") | pynini.accep(" ")).optimize()

        alphas = (latin_letters | devanagari_word).optimize()
        segment = (alphas | num_graph | symbols_graph).optimize()

        serial_core = segment + pynini.closure(delimiter + segment, 1)
        serial_core = serial_core.optimize()

        serial_graph = serial_core

        all_alphas = pynini.union(NEMO_ALPHA, devanagari_chars).optimize()

        insert_space_alpha_digit = pynini.cdrewrite(pynutil.insert(" "), all_alphas, any_digit, NEMO_SIGMA)
        insert_space_digit_alpha = pynini.cdrewrite(pynutil.insert(" "), any_digit, all_alphas, NEMO_SIGMA)
        space_inserter = pynini.compose(insert_space_alpha_digit, insert_space_digit_alpha).optimize()

        glued_serial = pynini.compose(space_inserter, serial_core).optimize()
        serial_graph = pynini.union(serial_graph, glued_serial).optimize()

        power_special = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/serial/power_special.tsv")), -1.0
        ).optimize()

        power_generic = pynutil.add_weight(
            (pynutil.delete("^") + pynutil.insert(" टु द पावर ") + num_graph), 1.0
        ).optimize()

        power_suffix = pynini.union(power_special, power_generic).optimize()
        power_graph = num_graph + power_suffix
        serial_graph = pynini.union(serial_graph, power_graph).optimize()

        serial_graph = pynini.compose(pynini.closure(NEMO_NOT_SPACE, 2), serial_graph).optimize()

        pure_word_slash = pynini.closure(NEMO_ALPHA, 1) + pynini.accep("/") + pynini.closure(NEMO_ALPHA, 1)

        dimension_pattern = (
            pynini.closure(any_digit, 1) + (pynini.accep("x") | pynini.accep("X")) + pynini.closure(any_digit, 1)
        )

        _opt_delim = pynini.closure(pynini.accep("-") | pynini.accep(" "), 0, 1)
        latin_alphanum = (pynini.closure(NEMO_ALPHA, 1) + _opt_delim + pynini.closure(any_digit, 1)) | (
            pynini.closure(any_digit, 1) + _opt_delim + pynini.closure(NEMO_ALPHA, 1)
        )

        ordinal_suffixes = pynini.project(
            pynini.union(
                pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv")),
                pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv")),
            ),
            "input",
        ).optimize()
        ordinal_pattern = pynini.closure(any_digit, 1) + ordinal_suffixes

        date_year_suffix = pynini.project(
            pynini.string_file(get_abs_path("data/date/year_suffix.tsv")),
            "input",
        ).optimize()
        date_suffixes = pynini.project(
            pynini.string_file(get_abs_path("data/date/suffixes.tsv")),
            "input",
        ).optimize()
        date_pattern = (
            pynini.closure(any_digit, 1)
            + pynini.closure(pynini.accep("-") + pynini.closure(any_digit, 1), 0)
            + pynini.accep(" ")
            + pynini.union(date_year_suffix, date_suffixes)
        )

        exclusions = pure_word_slash | dimension_pattern | latin_alphanum | ordinal_pattern | date_pattern
        accepted_inputs = pynini.difference(NEMO_SIGMA, exclusions).optimize()

        serial_graph = pynini.compose(accepted_inputs, serial_graph).optimize()

        self.graph = serial_graph.optimize()
        graph = pynutil.insert('name: "') + convert_space(self.graph).optimize() + pynutil.insert('"')
        self.fst = graph.optimize()
