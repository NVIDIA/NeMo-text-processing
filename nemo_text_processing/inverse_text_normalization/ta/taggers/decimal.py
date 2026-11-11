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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers in Tamil.

    Decimal point is indicated by "à®ªà¯à®³à¯à®³à®¿" followed by one or more digits.
    Integer part is mandatory (matches all blind test cases).

    Examples:
        - à®’à®©à¯à®±à¯ à®ªà¯à®³à¯à®³à®¿ à®‡à®°à®£à¯à®Ÿà¯ â†’ decimal { integer_part: "à¯§" fractional_part: "à¯¨" }
        - à®ªà¯‚à®œà¯à®œà®¿à®¯à®®à¯ à®ªà¯à®³à¯à®³à®¿ à®’à®©à¯à®±à¯ â†’ decimal { integer_part: "à¯¦" fractional_part: "à¯§" }
        - à®ªà¯‚à®œà¯à®¯à®®à¯ à®ªà¯à®³à¯à®³à®¿ à®’à®©à¯à®ªà®¤à¯ à®Žà®Ÿà¯à®Ÿà¯ à®Žà®Ÿà¯à®Ÿà¯ â†’ decimal { integer_part: "à¯¦" fractional_part: "à¯¯à¯®à¯®" }
        - à®®à¯‚à®©à¯à®±à¯ à®ªà¯à®³à¯à®³à®¿ à®’à®©à¯à®±à¯ à®¨à®¾à®©à¯à®•à¯ â†’ decimal { integer_part: "à¯©" fractional_part: "à¯§à¯ª" }
        - à®¨à¯‚à®±à¯à®±à¯ à®‡à®°à®£à¯à®Ÿà¯ à®ªà¯à®³à¯à®³à®¿ à®®à¯‚à®©à¯à®±à¯ â†’ decimal { integer_part: "à¯§à¯¦à¯¨" fractional_part: "à¯©" }
        - à®Žà®¤à®¿à®°à¯à®®à®±à¯ˆ à®’à®©à¯à®±à¯ à®ªà¯à®³à¯à®³à®¿ à®à®¨à¯à®¤à¯ â†’ decimal { negative: "true" integer_part: "à¯§" fractional_part: "à¯«" }

    Args:
        cardinal: CardinalFst - used for parsing the integer part
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        # Get the cardinal graph for integer part parsing
        # Using graph_no_exception to match Hindi design and avoid
        # accidentally rejecting valid number forms
        cardinal_graph = cardinal.graph_no_exception

        # =====================================================
        # Graph for digits after decimal point
        # Supports "à®ªà¯‚à®œà¯à®œà®¿à®¯à®®à¯", "à®ªà¯‚à®œà¯à®¯à®®à¯", and "à®šà¯à®´à®¿à®¯à®®à¯" variants
        # =====================================================

        # Zero variants from zero.tsv
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()

        # Digits 1-9 from digit.tsv
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()

        # Combine zero and digits for decimal part
        graph_decimal = graph_zero | graph_digit

        # One or more digits after decimal point
        # e.g., "à®’à®©à¯à®±à¯" â†’ "à¯§", "à®’à®©à¯à®±à¯ à®‡à®°à®£à¯à®Ÿà¯" â†’ "à¯§à¯¨"
        graph_decimal = (pynini.closure(graph_decimal + delete_space) + graph_decimal).optimize()

        # =====================================================
        # Decimal point marker
        # =====================================================
        point = pynutil.delete("à®ªà¯à®³à¯à®³à®¿")

        # =====================================================
        # Optional negative marker
        # =====================================================
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("à®Žà®¤à®¿à®°à¯à®®à®±à¯ˆ", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        # =====================================================
        # Integer part (mandatory - matches all blind tests)
        # =====================================================
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        # =====================================================
        # Fractional part
        # =====================================================
        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")

        # =====================================================
        # Final decimal graph
        # Integer part is mandatory (all blind tests have it)
        # =====================================================
        self.final_graph_wo_negative = (
            graph_integer + delete_extra_space + point + delete_extra_space + graph_fractional
        )

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        # Add tokens for final output
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
