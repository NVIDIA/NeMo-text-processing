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
    GraphFst,
    delete_space,
    insert_space,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    NEMO_CHAR,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


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
        non_space_char = pynini.difference(NEMO_CHAR, pynini.union(convertible_char, NEMO_WHITE_SPACE))
        
        # Character-level processor:
        # - Convertible chars -> add space before and after, then convert
        # - Spaces -> keep as single space
        # - Commas -> add space before and after to separate from surrounding words
        # - Other chars -> keep as-is
        
        # For comma, add space before and after it
        comma_processor = insert_space + pynini.accep(",") + insert_space
        
        # For other non-space, non-comma chars, keep as-is
        other_char = pynini.difference(non_space_char, pynini.accep(","))
        
        char_processor = (
            insert_space + pynini.compose(convertible_char, char_to_word) + insert_space
        ) | pynini.accep(NEMO_SPACE) | comma_processor | other_char
        
        # Process entire string character by character
        # This creates a graph that converts all digits/special chars and keeps everything else
        full_string_processor = pynini.closure(char_processor, 1)
        
        # Now we need to only apply this when address context is present
        # Create patterns that match strings containing address keywords
        any_char = pynini.union(NEMO_CHAR, NEMO_WHITE_SPACE)
        
        # Pattern: anything + address keyword + anything
        # This matches any string that contains at least one address context keyword
        has_address_keyword = (
            pynini.closure(any_char) +
            address_keywords +
            pynini.closure(any_char)
        )
        
        # Require that the string contains at least one digit
        has_digit = (
            pynini.closure(any_char) +
            pynini.union(single_digit) +
            pynini.closure(any_char)
        )
        
        # IMPORTANT: Exclude long digit sequences that look like telephone numbers
        # Telephone numbers typically have 10+ consecutive digits
        # Addresses typically have shorter digit sequences (1-5 digits)
        long_digit_sequence = pynini.closure(convertible_char, 10)  # 10+ digits
        has_long_digits = (
            pynini.closure(any_char) +
            long_digit_sequence +
            pynini.closure(any_char)
        )
        
        # Input must:
        # 1. Have address keyword (Hindi or English)
        # 2. Have digits
        # 3. NOT have long digit sequences (phone numbers)
        input_pattern = pynini.intersect(
            pynini.intersect(has_address_keyword, has_digit),
            pynini.difference(pynini.union(any_char).closure(), has_long_digits)
        )
        
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
        quarterly_units_graph = pynini.string_file(get_abs_path("data/measure/quarterly_units.tsv"))

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Define the quarterly measurements
        quarter = pynini.string_map(
            [
                (".५", "साढ़े"),
                ("१.५", "डेढ़"),
                ("२.५", "ढाई"),
            ]
        )
        quarter_graph = pynutil.insert("integer_part: \"") + quarter + pynutil.insert("\"")

        # Define the unit handling
        unit = pynutil.insert(" units: \"") + unit_graph + pynutil.insert("\" ")
        units = pynutil.insert(" units: \"") + quarterly_units_graph + pynutil.insert("\" ")

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

        dedh_dhai = pynini.string_map([("१.५", "डेढ़"), ("२.५", "ढाई")])
        dedh_dhai_graph = pynutil.insert("integer: \"") + dedh_dhai + pynutil.insert("\"")

        savva_numbers = cardinal_graph + pynini.cross(".२५", "")
        savva_graph = pynutil.insert("integer: \"सवा ") + savva_numbers + pynutil.insert("\"")

        sadhe_numbers = cardinal_graph + pynini.cross(".५", "")
        sadhe_graph = pynutil.insert("integer: \"साढ़े ") + sadhe_numbers + pynutil.insert("\"")

        paune = pynini.string_file(get_abs_path("data/whitelist/paune_mappings.tsv"))
        paune_numbers = paune + pynini.cross(".७५", "")
        paune_graph = pynutil.insert("integer: \"पौने ") + paune_numbers + pynutil.insert("\"")

        graph_dedh_dhai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + dedh_dhai_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        graph_savva = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + savva_graph
            + pynutil.insert(" }")
            + delete_space
            + units
        )

        graph_sadhe = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + sadhe_graph
            + pynutil.insert(" }")
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
            + pynutil.insert(" }")
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
            + pynutil.insert(" units: \"")
            + symbol_graph
            + pynutil.insert("\" ")
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
            pynutil.add_weight(graph_decimal, 0.01)
            | pynutil.add_weight(graph_cardinal, 0.01)
            | pynutil.add_weight(graph_exceptions, 0.01)
            | pynutil.add_weight(graph_dedh_dhai, 0.001)
            | pynutil.add_weight(graph_savva, 0.005)
            | pynutil.add_weight(graph_sadhe, 0.005)
            | pynutil.add_weight(graph_paune, -0.2)
            | address_graph  # Include address graph
        )
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
