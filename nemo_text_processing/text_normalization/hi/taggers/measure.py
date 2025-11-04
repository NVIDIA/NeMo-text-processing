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
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

HI_POINT_FIVE = ".५"  # .5
HI_ONE_POINT_FIVE = "१.५"  # 1.5
HI_TWO_POINT_FIVE = "२.५"  # 2.5
HI_DECIMAL_25 = ".२५"  # .25
HI_DECIMAL_75 = ".७५"  # .75

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

    def get_address_graph(self, cardinal: GraphFst):
        """
        Address tagger that converts digits/hyphens/slashes character-by-character
        when address context keywords are present, keeping all surrounding text.
        
        Examples:
            "७०० ओक स्ट्रीट" -> "सात शून्य शून्य ओक स्ट्रीट"
            "६६-४ पार्क रोड" -> "छह छह हाइफ़न चार पार्क रोड"
        """
        # Load character mappings
        char_to_word = (
            pynini.string_file(get_abs_path("data/address/address_digits.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )
        
        # Load address context keywords (Hindi and English)
        address_keywords_hi = pynini.string_file(get_abs_path("data/address/context.tsv"))
        address_keywords_en = pynini.string_file(get_abs_path("data/address/en_context.tsv"))
        address_keywords = address_keywords_hi | address_keywords_en
        
        # Define character sets
        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT
        special_chars = pynini.union("-", "/")
        convertible_char = single_digit | special_chars
        
        # Non-convertible characters (everything else except space)
        non_space_char = pynini.difference(NEMO_CHAR, pynini.union(NEMO_WHITE_SPACE, convertible_char, pynini.accep(",")))
        
        # Character-level processor:
        # - Convertible chars -> add space before and after, then convert
        # - Spaces -> keep as single space
        # - Commas -> add space before and after to separate from surrounding words
        # - Other chars -> keep as-is
        
        # For comma, add space before and after it
        comma_processor = insert_space + pynini.accep(",") + insert_space
        
        # For other non-space, non-comma chars, keep as-is
        # other_char = pynini.difference(non_space_char, pynini.accep(","))
        
        char_processor = (
            insert_space + pynini.compose(convertible_char, char_to_word) + insert_space
        ) | pynini.accep(NEMO_SPACE) | comma_processor | non_space_char
        
        # Process entire string character by character
        # This creates a graph that converts all digits/special chars and keeps everything else
        full_string_processor = pynini.closure(char_processor, 1)
        
        # Now we need to only apply this when address context is present
        # Create patterns that match strings containing address keywords
        any_char = pynini.union(NEMO_CHAR, NEMO_WHITE_SPACE)
        
        # # Pattern: anything + address keyword + anything
        # # This matches any string that contains at least one address context keyword
        # has_address_keyword = (
        #     pynini.closure(any_char) +
        #     address_keywords +
        #     pynini.closure(any_char)
        # )

        # Define word boundaries: space, comma, Hindi fullstop
        word_boundary = pynini.union(NEMO_WHITE_SPACE, pynini.accep(","), pynini.accep("।"))
        
        # Define a word as a sequence of non-boundary characters
        non_boundary_char = pynini.difference(NEMO_CHAR, word_boundary)
        word = pynini.closure(non_boundary_char, 1)
        
        # Word with optional boundaries after it (allows multiple: ", " = comma + space)
        word_with_boundary = word + pynini.closure(word_boundary)
        
        # Up to 4 words (for the window)
        up_to_4_words = pynini.closure(word_with_boundary, 0, 5)
        
        # Match context word with word boundaries to prevent substring matching
        # Three cases for the context word with boundaries:
        # 1. Start of string: keyword + boundaries (or end of string)
        # 2. Middle: boundaries + keyword + boundaries  
        # 3. End: boundaries + keyword (at end of string)
        context_at_start = address_keywords + pynini.closure(word_boundary)
        context_in_middle = pynini.closure(word_boundary, 1) + address_keywords + pynini.closure(word_boundary)
        context_at_end = pynini.closure(word_boundary, 1) + address_keywords
        
        # Pattern that matches strings with context word within a 4-word window
        # Case 1: Context at start - match: context + up to 4 words after
        pattern1 = context_at_start + up_to_4_words
        
        # Case 2: Context in middle - match: up to 4 words before + context + up to 4 words after
        pattern2 = up_to_4_words + context_in_middle + up_to_4_words
        
        # Case 3: Context at end - match: up to 4 words before + context
        pattern3 = up_to_4_words + context_at_end
        
        # Combine all patterns
        input_pattern = pattern1 | pattern2 | pattern3
        
        # Digit detection (commented out for now - to be checked later)
        # has_digit = (
        #     pynini.closure(any_char) +
        #     pynini.union(single_digit) +
        #     pynini.closure(any_char)
        # )
        # input_pattern = pynini.intersect(input_pattern, has_digit)
        
        # Apply the character processor to inputs matching the pattern
        address_graph = pynini.compose(input_pattern, full_string_processor)
        
        # Wrap as measure with "address" unit
        graph = (
            pynutil.insert('units: "address" cardinal { integer: "') +
            address_graph +
            pynutil.insert('" } preserve_order: true')
        )
        
        # Lower priority than telephone to avoid conflicts
        # Telephone has weights 0.7-1.0, so use 1.05 to be lower priority
        return pynutil.add_weight(graph, 1.05).optimize()

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
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

        # Define the unit handling
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

        # Handling symbols like x, X, *
        symbol_graph = pynini.string_map(
            [
                ("x", "बाई"),
                ("X", "बाई"),
                ("*", "बाई"),
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

        # Get the address graph for digit-by-digit conversion in address contexts
        address_graph = self.get_address_graph(cardinal)
        
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
