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

    def get_address_graph(self, cardinal: GraphFst, ordinal: GraphFst, input_case: str):
        """
        Address tagger that converts digits/hyphens/slashes character-by-character
        when address context keywords are present, keeping all surrounding text.
        English words are converted to Hindi transliterations.
        Ordinals (e.g., 1st, 2nd, 3rd) are transliterated to Hindi.
        
        Examples:
            "७०० ओक स्ट्रीट" -> "सात शून्य शून्य ओक स्ट्रीट"
            "६६-४ पार्क रोड" -> "छह छह हाइफ़न चार पार्क रोड"
            "७०० ओक Street" -> "सात शून्य शून्य ओक स्ट्रीट"
            "यूनिट ३ 1st फ्लोर" -> "यूनिट तीन फ़र्स्ट फ्लोर"
        """
        # Use the ordinal exceptions graph from OrdinalFst for addresses
        # This includes both English-style ordinals (1st, 2nd, 3rd) and Hindi-style ordinals (५वां, ६ठा)
        # Both should be read as complete ordinals in addresses, not digit-by-digit
        ordinal_graph = ordinal.exceptions_graph
        
        # Load character mappings
        char_to_word = (
            pynini.string_file(get_abs_path("data/address/address_digits.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )
        
        # Load letter transliterations (A-Z, a-z)
        letter_to_word = pynini.string_file(get_abs_path("data/address/letters.tsv"))
        
        # Load address context keywords (Hindi only for context detection)
        address_keywords_hi = pynini.string_file(get_abs_path("data/address/context.tsv"))
        
        # Load English-to-Hindi mapping for transliteration with case-insensitive support
        en_to_hi_mapping = load_labels(get_abs_path("data/address/en_to_hi_mapping.tsv"))
        
        # Extract English context keywords from the mapping (first column)
        en_context_words = []
        if input_case == INPUT_LOWER_CASED:
            # For lower_cased input, just use lowercase mappings
            en_to_hi_mapping_expanded = [[x.lower(), y] for x, y in en_to_hi_mapping]
            en_context_words = [x.lower() for x, _ in en_to_hi_mapping]
        else:
            # For cased input, generate both lowercase and capitalized versions for case-insensitive matching
            expanded_mapping = []
            for x, y in en_to_hi_mapping:
                expanded_mapping.append([x, y])  # lowercase version
                en_context_words.append(x)  # Add to context words
                # Add capitalized version (first letter uppercase)
                if x and x[0].isalpha():
                    capitalized = x[0].upper() + x[1:]
                    if capitalized != x:  # avoid duplicates
                        expanded_mapping.append([capitalized, y])
                        en_context_words.append(capitalized)  # Add capitalized to context
            en_to_hi_mapping_expanded = expanded_mapping
        
        en_to_hi_map = pynini.string_map(en_to_hi_mapping_expanded)
        
        # For context detection, use both Hindi words and English words extracted from mapping
        address_keywords_en = pynini.string_map([[word, word] for word in en_context_words])
        address_keywords = address_keywords_hi | address_keywords_en
        
        # Define character sets
        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT
        special_chars = pynini.union("-", "/")
        
        # Define single English letters (A-Z, a-z) - to be handled separately from words
        single_letter = pynini.union(
            *[pynini.accep(chr(i)) for i in range(ord('a'), ord('z')+1)]
        ) | pynini.union(
            *[pynini.accep(chr(i)) for i in range(ord('A'), ord('Z')+1)]
        )
        
        convertible_char = single_digit | special_chars
        
        # Non-convertible characters (everything else except space, letters, and convertibles)
        non_space_char = pynini.difference(
            NEMO_CHAR, 
            pynini.union(NEMO_WHITE_SPACE, convertible_char, pynini.accep(","), single_letter)
        )
        
        # Character-level processor:
        # - Ordinals (1st, 2nd, etc.) -> convert to Hindi transliteration (highest priority)
        # - English words -> convert to Hindi transliteration with spaces
        # - Single letters (A, B, C, etc.) -> convert to Hindi transliteration with spaces
        # - Convertible chars (digits/hyphens) -> add space before and after, then convert
        # - Spaces -> keep as single space
        # - Commas -> add space before and after to separate from surrounding words
        # - Other chars -> keep as-is
        
        # For comma, add space before and after it
        comma_processor = insert_space + pynini.accep(",") + insert_space
        
        # Ordinal processor with highest priority
        # The ordinal graph handles both English and Hindi digit ordinals
        ordinal_processor = pynutil.add_weight(
            insert_space + ordinal_graph + insert_space,
            -20.0  # Highest priority (most negative weight)
        )
        
        # English word processor with priority: match English word and convert to Hindi
        # Use weight to prefer longer matches (English words) over character matches
        english_word_processor = pynutil.add_weight(
            insert_space + en_to_hi_map + insert_space,
            -10.0  # Very high priority to ensure words match before letters
        )
        
        # Single letter processor - handles isolated English letters
        # Lower priority than words and digits - only match when nothing else matches
        letter_processor = pynutil.add_weight(
            insert_space + pynini.compose(single_letter, letter_to_word) + insert_space,
            0.5  # Lower priority than words and digits, higher than other chars
        )
        
        # Character-level processors for digits/special chars
        digit_char_processor = pynutil.add_weight(
            insert_space + pynini.compose(convertible_char, char_to_word) + insert_space,
            0.0  # Neutral weight
        )
        
        # Other single characters (lower priority)
        other_char_processor = pynutil.add_weight(
            non_space_char,
            0.1  # Positive weight = lower priority
        )
        
        # Combined processor: Ordinals > English words > Single letters > Digits > Other chars
        token_processor = (
            ordinal_processor
            | english_word_processor
            | letter_processor
            | digit_char_processor
            | pynini.accep(NEMO_SPACE)
            | comma_processor
            | other_char_processor
        )
        
        # Process entire string token by token
        full_string_processor = pynini.closure(token_processor, 1)
        
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
        
        # Word with optio
        # 
        # nal boundaries after it (allows multiple: ", " = comma + space)
        word_with_boundary = word + pynini.closure(word_boundary)
        
        # Up to 4 words (for the window)
        up_to_4_words = pynini.closure(word_with_boundary, 0, 5)
        
        # Match context word with word boundaries to prevent substring matching
        # Three cases for the context word with boundaries:
        # 1. Start of string: keyword + boundaries (or end of string)
        # 2. Middle: boundaries + keyword + boundaries  
        # 3. End: boundaries + keyword (at end of string)
        context_at_start = address_keywords + pynini.closure(word_boundary, 1)
        context_in_middle = pynini.closure(word_boundary, 1) + address_keywords + pynini.closure(word_boundary, 1)
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
        address_graph = self.get_address_graph(cardinal, ordinal, input_case)
        
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
