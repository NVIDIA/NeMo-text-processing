# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
    HI_DEDH,
    HI_DHAI,
    HI_PAUNE,
    HI_SADHE,
    HI_SAVVA,
    INPUT_LOWER_CASED,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path, load_labels

HI_POINT_FIVE = ".५"
HI_ONE_POINT_FIVE = "१.५"
HI_TWO_POINT_FIVE = "२.५"
HI_DECIMAL_25 = ".२५"
HI_DECIMAL_75 = ".७५"
HI_BY = "बाई"

LOWERCASE_X = "x"
UPPERCASE_X = "X"
ASTERISK = "*"
HYPHEN = "-"
SLASH = "/"
COMMA = ","
PERIOD = "."
HI_PERIOD = "।"

digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
teens_and_ties = pynutil.add_weight(teens_ties, -0.1)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -१२kg -> measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" }
        -१२.२kg -> measure { decimal { negative: "true"  integer_part: "बारह"  fractional_part: "दो"} units: "किलोग्राम" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def get_address_graph(self, ordinal: GraphFst, input_case: str):
        """
        Address tagger that converts digits/hyphens/slashes character-by-character
        when address context keywords are present.
        English words and ordinals are converted to Hindi transliterations.
        
        Examples:
            "७०० ओक स्ट्रीट" -> "सात शून्य शून्य ओक स्ट्रीट"
            "६६-४ पार्क रोड" -> "छह छह हाइफ़न चार पार्क रोड"
        """
        ordinal_graph = ordinal.graph
        char_to_word = (
            digit
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
            | pynini.string_file(get_abs_path("data/address/special_characters.tsv"))
            | pynini.string_file(get_abs_path("data/telephone/number.tsv"))
        )
        letter_to_word = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        address_keywords_hi = pynini.string_file(get_abs_path("data/address/context.tsv"))
        en_to_hi_mapping = load_labels(get_abs_path("data/address/en_to_hi_mapping.tsv"))
        en_context_words = []
        if input_case == INPUT_LOWER_CASED:
            en_to_hi_mapping_expanded = [[x.lower(), y] for x, y in en_to_hi_mapping]
            en_context_words = [x.lower() for x, _ in en_to_hi_mapping]
        else:
            expanded_mapping = []
            for x, y in en_to_hi_mapping:
                expanded_mapping.append([x, y])
                en_context_words.append(x)
                if x and x[0].isalpha():
                    capitalized = x[0].upper() + x[1:]
                    if capitalized != x:
                        expanded_mapping.append([capitalized, y])
                        en_context_words.append(capitalized)
            en_to_hi_mapping_expanded = expanded_mapping
        en_to_hi_map = pynini.string_map(en_to_hi_mapping_expanded)

        address_keywords_en = pynini.string_map([[word, word] for word in en_context_words])
        address_keywords = address_keywords_hi | address_keywords_en
        
        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT
        special_chars = pynini.union(HYPHEN, SLASH)
        single_letter = pynini.project(letter_to_word, "input")
        convertible_char = single_digit | special_chars | single_letter
        non_space_char = pynini.difference(
            NEMO_CHAR, 
            pynini.union(NEMO_WHITE_SPACE, convertible_char, pynini.accep(COMMA))
        )
        comma_processor = insert_space + pynini.accep(COMMA) + insert_space
        ordinal_processor = pynutil.add_weight(insert_space + ordinal_graph + insert_space, -5.0)
        english_word_processor = pynutil.add_weight(insert_space + en_to_hi_map + insert_space, -3.0)
        letter_processor = pynutil.add_weight(insert_space + pynini.compose(single_letter, letter_to_word) + insert_space, 0.5)
        digit_char_processor = pynutil.add_weight(
            insert_space + pynini.compose(convertible_char, char_to_word) + insert_space,
            0.0
        )
        other_char_processor = pynutil.add_weight(
            non_space_char,
            0.1
        )
        
        token_processor = (
            ordinal_processor
            | english_word_processor
            | letter_processor
            | digit_char_processor
            | pynini.accep(NEMO_SPACE)
            | comma_processor
            | other_char_processor
        )
        full_string_processor = pynini.closure(token_processor, 1)
        word_boundary = pynini.union(NEMO_WHITE_SPACE, pynini.accep(COMMA), pynini.accep(HI_PERIOD), pynini.accep(PERIOD))
        non_boundary_char = pynini.difference(NEMO_CHAR, word_boundary)
        word = pynini.closure(non_boundary_char, 1)
        word_with_boundary = word + pynini.closure(word_boundary)
        window = pynini.closure(word_with_boundary, 0, 5)
        boundary = pynini.closure(word_boundary, 1)
        input_pattern = pynini.union(
            address_keywords + boundary + window,
            window + boundary + address_keywords + pynini.closure(boundary + window, 0, 1)
        )
        address_graph = pynini.compose(input_pattern, full_string_processor)
        graph = (
            pynutil.insert('units: "address" cardinal { integer: "') +
            address_graph +
            pynutil.insert('" } preserve_order: true')
        )
        return pynutil.add_weight(graph, 1.05).optimize()

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, ordinal: GraphFst, input_case: str):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = (
            cardinal.zero
            | cardinal.digit
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
            | cardinal.graph_lakhs
            | cardinal.graph_ten_lakhs
        )
        point = pynutil.delete(".")
        decimal_integers = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        decimal_graph = decimal_integers + point + insert_space + decimal.graph_fractional

        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        # Load quarterly units from separate files: map (FST) and list (FSA)
        quarterly_units_map = pynini.string_file(get_abs_path("data/measure/quarterly_units_map.tsv"))
        quarterly_units_list = pynini.string_file(get_abs_path("data/measure/quarterly_units_list.tsv"))
        quarterly_units_graph = pynini.union(quarterly_units_map, quarterly_units_list)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Define the quarterly measurements
        quarter = pynini.string_map(
            [
                (HI_POINT_FIVE, HI_SADHE),
                (HI_ONE_POINT_FIVE, HI_DEDH),
                (HI_TWO_POINT_FIVE, HI_DHAI),
            ]
        )
        quarter_graph = pynutil.insert("integer_part: \"") + quarter + pynutil.insert("\"")

        unit = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + unit_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )
        units = (
            pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + quarterly_units_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        symbol_graph = pynini.string_map(
            [
                (LOWERCASE_X, HI_BY),
                (UPPERCASE_X, HI_BY),
                (ASTERISK, HI_BY),
            ]
        )

        graph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        dedh_dhai = pynini.string_map([(HI_ONE_POINT_FIVE, HI_DEDH), (HI_TWO_POINT_FIVE, HI_DHAI)])
        dedh_dhai_graph = pynutil.insert("integer: \"") + dedh_dhai + pynutil.insert("\"")

        savva_numbers = cardinal_graph + pynini.cross(HI_DECIMAL_25, "")
        savva_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_SAVVA)
            + pynutil.insert(NEMO_SPACE)
            + savva_numbers
            + pynutil.insert("\"")
        )

        sadhe_numbers = cardinal_graph + pynini.cross(HI_POINT_FIVE, "")
        sadhe_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_SADHE)
            + pynutil.insert(NEMO_SPACE)
            + sadhe_numbers
            + pynutil.insert("\"")
        )

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(HI_DECIMAL_75, "")
        paune_graph = (
            pynutil.insert("integer: \"")
            + pynutil.insert(HI_PAUNE)
            + pynutil.insert(NEMO_SPACE)
            + paune_numbers
            + pynutil.insert("\"")
        )

        graph_dedh_dhai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + dedh_dhai_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_savva = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + savva_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_sadhe = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sadhe_graph
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + units
        )

        graph_paune = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + paune_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("}")
            + delete_space
            + unit
        )

        # Handling cardinal clubbed with symbol as single token
        graph_exceptions = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("units: \"")
            + symbol_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("} }")
            + insert_space
            + pynutil.insert("tokens { cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        address_graph = self.get_address_graph(ordinal, input_case)
        
        graph = (
            pynutil.add_weight(graph_decimal, 0.1)
            | pynutil.add_weight(graph_cardinal, 0.1)
            | pynutil.add_weight(graph_exceptions, 0.1)
            | pynutil.add_weight(graph_dedh_dhai, -0.2)
            | pynutil.add_weight(graph_savva, -0.1)
            | pynutil.add_weight(graph_sadhe, -0.1)
            | pynutil.add_weight(graph_paune, -0.5)
            | address_graph
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
