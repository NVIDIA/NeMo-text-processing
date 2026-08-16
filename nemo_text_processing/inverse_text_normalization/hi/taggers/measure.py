# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ऋण बारह किलोग्राम -> measure { decimal { negative: "true"  integer_part: "१२"  fractional_part: "५०"} units: "kg" }
        e.g. ऋण बारह किलोग्राम -> measure { cardinal { negative: "true"  integer_part: "१२"} units: "kg" }
        e.g. सात शून्य शून्य ओक स्ट्रीट -> measure { units: "address" cardinal { integer: "७०० ओक स्ट्रीट" } preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ऋण", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        measurements_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.measurements = pynutil.insert("units: \"") + measurements_graph + pynutil.insert("\" ")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer_paune = pynutil.insert("integer_part: \"") + paune_graph + pynutil.insert("\"")

        graph_saade_single_digit = pynutil.add_weight(
            pynutil.delete("साढ़े")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"५\""),
            0.1,
        )
        graph_sava_single_digit = pynutil.add_weight(
            pynutil.delete("सवा")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"२५\""),
            0.1,
        )
        graph_paune_single_digit = pynutil.add_weight(
            pynutil.delete("पौने")
            + delete_space
            + graph_integer_paune
            + delete_space
            + pynutil.insert(" fractional_part: \"७५\""),
            1,
        )
        graph_dedh_single_digit = pynutil.add_weight(
            pynini.union(pynutil.delete("डेढ़") | pynutil.delete("डेढ़"))
            + delete_space
            + pynutil.insert("integer_part: \"१\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"५\""),
            0.1,
        )
        graph_dhaai_single_digit = pynutil.add_weight(
            pynutil.delete("ढाई")
            + delete_space
            + pynutil.insert("integer_part: \"२\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"५\""),
            1,
        )

        graph_exceptions = (
            graph_saade_single_digit
            | graph_sava_single_digit
            | graph_paune_single_digit
            | graph_dedh_single_digit
            | graph_dhaai_single_digit
        )

        graph_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_measurements |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_quarterly_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + graph_exceptions
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_exception_bai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + delete_space
            + pynini.cross("बाई", "x")
            + delete_space
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space + self.measurements)
        )

        # Shared digit word -> Devanagari digit mapping
        num_word = (
            (
                pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
                | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
                | pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv"))
                | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv"))
            )
            .invert()
            .optimize()
        )

        delete_one_space = pynutil.delete(" ")

        # Structured address: state/city + pincode
        states = pynini.string_file(get_abs_path("data/address/states.tsv"))
        cities = pynini.string_file(get_abs_path("data/address/cities.tsv"))
        state_city_names = pynini.union(states, cities).optimize()

        pincode = num_word + pynini.closure(delete_one_space + num_word, 5, 5)

        structured_pattern = (
            state_city_names
            + pynini.closure(pynini.accep(",") + pynini.accep(" ") + state_city_names, 0, 1)
            + pynini.accep(" ")
            + pincode
        ).optimize()

        structured_address_graph = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + convert_space(structured_pattern)
            + pynutil.insert('" } preserve_order: true')
        )
        structured_address_graph = pynutil.add_weight(structured_address_graph, 1.0).optimize()

        # Address: digit/special/ordinal conversion with context keywords
        special_word = pynini.string_file(get_abs_path("data/address/special_characters.tsv"))
        ordinal_word = pynini.string_file(get_abs_path("data/address/ordinals.tsv"))
        context_keywords_fsa = pynini.string_file(get_abs_path("data/address/context_cues.tsv"))

        digit_passthrough = pynini.string_file(get_abs_path("data/address/digit_passthrough.tsv")).optimize()
        digit_unit = pynini.union(num_word, digit_passthrough).optimize()

        all_digit_inputs = pynini.project(digit_unit, "input").optimize()
        all_ordinal_inputs = pynini.project(ordinal_word, "input").optimize()

        non_space_non_comma = pynini.difference(
            NEMO_CHAR, pynini.union(NEMO_WHITE_SPACE, pynini.accep(","))
        ).optimize()
        any_word = pynini.closure(non_space_non_comma, 1).optimize()

        text_word = pynini.difference(any_word, pynini.union(all_digit_inputs, all_ordinal_inputs)).optimize()

        digit_block = digit_unit + pynini.closure(pynutil.add_weight(delete_one_space + digit_unit, -1.0))

        connector = delete_one_space + special_word + delete_one_space

        matchable = pynini.union(
            pynutil.add_weight(digit_block, -0.1),
            pynutil.add_weight(ordinal_word, -0.2),
            pynutil.add_weight(text_word, 0.1),
        ).optimize()

        chain = matchable + pynini.closure(pynutil.add_weight(connector + matchable, -0.5))

        opt_comma = pynini.closure(pynini.accep(","), 0, 1)
        element = chain + opt_comma
        address_content = element + pynini.closure(pynini.accep(" ") + element)

        # Context detection: keyword must appear as a complete word in the input
        any_char = pynini.union(
            pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE),
            NEMO_WHITE_SPACE,
        ).optimize()
        sigma_star = pynini.closure(any_char).optimize()

        word_sep = pynini.union(pynini.accep(" "), pynini.accep(",")).optimize()

        input_pattern = pynini.union(
            context_keywords_fsa + word_sep + sigma_star,
            sigma_star + pynini.accep(" ") + context_keywords_fsa,
            sigma_star + pynini.accep(" ") + context_keywords_fsa + word_sep + sigma_star,
            context_keywords_fsa,
        ).optimize()

        address_graph = pynini.compose(input_pattern, address_content).optimize()

        address_graph = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + convert_space(address_graph)
            + pynutil.insert('" } preserve_order: true')
        )
        address_graph = pynutil.add_weight(address_graph, 1.05).optimize()

        graph = (
            graph_measurements
            | graph_quarterly_measurements
            | graph_exception_bai
            | address_graph
            | structured_address_graph
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
