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
# WITHOUT WARRANTIES OR CONDIIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.kn.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_ALL_ZERO,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals in Kannada, e.g.
        -23 -> cardinal { negative: "true"  integer: "ಇಪ್ಪತ್ತಮೂರು" }
        123 -> cardinal { integer: "ನೂರ ಇಪ್ಪತ್ತಮೂರು" }
        500 -> cardinal { integer: "ಐನೂರು" }
        1001 -> cardinal { integer: "ಒಂದು ಸಾವಿರದ ಒಂದು" }
        
    Uses contracted hundred forms (ಐನೂರು, ಇನ್ನೂರು) and genitive connectors
    (ಸಾವಿರದ, ಲಕ್ಷದ, ಕೋಟಿಯ) when followed by non-zero remainders.
    
    Numbers above ಕೋಟಿ (10^7) use ಕೋಟಿ as base unit:
        10^9 -> ನೂರು ಕೋಟಿ (not ಅರಬ್)
        10^11 -> ಹತ್ತು ಸಾವಿರ ಕೋಟಿ (not ಖರಬ್)
    
    Coverage limit: Numbers up to 99 lakh crore (99,99,999 crore or ~10^14) are supported.
    Numbers with crore coefficients > 99,99,999 (i.e., requiring 8+ digit coefficient
    normalization) will fall through as raw text. This covers practical TTS use cases.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties_kn = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
        teens_ties = pynini.union(teens_ties_kn, teens_ties_en)
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties

        # Single digit graph for digit-by-digit reading
        single_digit_graph = digit | zero
        self.single_digits_graph = single_digit_graph + pynini.closure(insert_space + single_digit_graph)

        # Helper for deleting zeros
        delete_zero = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO), -0.1)

        # ============================================================
        # HUNDREDS (100-999)
        # Using contracted forms: ನೂರ/ನೂರು, ಐನೂರ/ಐನೂರು, ಒಂಬೈನೂರ/ಒಂಬೈನೂರು
        # ============================================================
        hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))
        self.hundreds = hundreds
        suffix_u = pynutil.insert("ು")

        # Standalone: 500 -> ಐನೂರು (with ು)
        graph_hundreds_standalone = hundreds + (delete_zero**2) + suffix_u
        # With digit: 501 -> ಐನೂರ ಒಂದು
        graph_hundreds_with_digit = hundreds + delete_zero + insert_space + digit
        # With teens/ties: 523 -> ಐನೂರ ಇಪ್ಪತ್ತಮೂರು
        graph_hundreds_with_teens = hundreds + insert_space + teens_ties

        graph_hundreds = graph_hundreds_standalone | graph_hundreds_with_digit | graph_hundreds_with_teens
        graph_hundreds.optimize()
        self.graph_hundreds = graph_hundreds

        # ============================================================
        # THOUSANDS (1000-99999) - ಸಾವಿರ/ಸಾವಿರದ
        # Standalone: ಸಾವಿರ | With remainder: ಸಾವಿರದ
        # ============================================================
        suffix_thousand_standalone = pynutil.insert(" ಸಾವಿರ")
        suffix_thousand_genitive = pynutil.insert(" ಸಾವಿರದ")

        # Standalone thousands (X000, XX000)
        graph_thousands_standalone = digit + (delete_zero**3) + suffix_thousand_standalone
        graph_ten_thousands_standalone = teens_and_ties + (delete_zero**3) + suffix_thousand_standalone

        # Thousands with remainder - use genitive ಸಾವಿರದ
        # X00Y, X0YZ, XYZW patterns
        graph_thousands_with_digit = digit + (delete_zero**2) + suffix_thousand_genitive + insert_space + digit
        graph_thousands_with_teens = digit + delete_zero + suffix_thousand_genitive + insert_space + teens_ties
        graph_thousands_with_hundreds = digit + suffix_thousand_genitive + insert_space + graph_hundreds

        graph_ten_thousands_with_digit = teens_and_ties + (delete_zero**2) + suffix_thousand_genitive + insert_space + digit
        graph_ten_thousands_with_teens = teens_and_ties + delete_zero + suffix_thousand_genitive + insert_space + teens_ties
        graph_ten_thousands_with_hundreds = teens_and_ties + suffix_thousand_genitive + insert_space + graph_hundreds

        graph_thousands = (
            graph_thousands_standalone
            | graph_thousands_with_digit
            | graph_thousands_with_teens
            | graph_thousands_with_hundreds
        )
        graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        graph_ten_thousands = (
            graph_ten_thousands_standalone
            | graph_ten_thousands_with_digit
            | graph_ten_thousands_with_teens
            | graph_ten_thousands_with_hundreds
        )
        graph_ten_thousands.optimize()
        self.graph_ten_thousands = graph_ten_thousands

        # ============================================================
        # LAKHS (100000-9999999) - ಲಕ್ಷ/ಲಕ್ಷದ
        # ============================================================
        suffix_lakh_standalone = pynutil.insert(" ಲಕ್ಷ")
        suffix_lakh_genitive = pynutil.insert(" ಲಕ್ಷದ")

        # Standalone lakhs
        graph_lakhs_standalone = digit + (delete_zero**5) + suffix_lakh_standalone
        graph_ten_lakhs_standalone = teens_and_ties + (delete_zero**5) + suffix_lakh_standalone

        # Lakhs with remainder
        graph_lakhs_with_digit = digit + (delete_zero**4) + suffix_lakh_genitive + insert_space + digit
        graph_lakhs_with_teens = digit + (delete_zero**3) + suffix_lakh_genitive + insert_space + teens_ties
        graph_lakhs_with_hundreds = digit + (delete_zero**2) + suffix_lakh_genitive + insert_space + graph_hundreds
        graph_lakhs_with_thousands = digit + delete_zero + suffix_lakh_genitive + insert_space + graph_thousands
        graph_lakhs_with_ten_thousands = digit + suffix_lakh_genitive + insert_space + graph_ten_thousands

        graph_ten_lakhs_with_digit = teens_and_ties + (delete_zero**4) + suffix_lakh_genitive + insert_space + digit
        graph_ten_lakhs_with_teens = teens_and_ties + (delete_zero**3) + suffix_lakh_genitive + insert_space + teens_ties
        graph_ten_lakhs_with_hundreds = teens_and_ties + (delete_zero**2) + suffix_lakh_genitive + insert_space + graph_hundreds
        graph_ten_lakhs_with_thousands = teens_and_ties + delete_zero + suffix_lakh_genitive + insert_space + graph_thousands
        graph_ten_lakhs_with_ten_thousands = teens_and_ties + suffix_lakh_genitive + insert_space + graph_ten_thousands

        graph_lakhs = (
            graph_lakhs_standalone
            | graph_lakhs_with_digit
            | graph_lakhs_with_teens
            | graph_lakhs_with_hundreds
            | graph_lakhs_with_thousands
            | graph_lakhs_with_ten_thousands
        )
        graph_lakhs.optimize()
        self.graph_lakhs = graph_lakhs

        graph_ten_lakhs = (
            graph_ten_lakhs_standalone
            | graph_ten_lakhs_with_digit
            | graph_ten_lakhs_with_teens
            | graph_ten_lakhs_with_hundreds
            | graph_ten_lakhs_with_thousands
            | graph_ten_lakhs_with_ten_thousands
        )
        graph_ten_lakhs.optimize()
        self.graph_ten_lakhs = graph_ten_lakhs

        # ============================================================
        # CRORES (10000000-999999999) - ಕೋಟಿ/ಕೋಟಿಯ
        # Note: ಕೋಟಿ ends in vowel, so genitive is ಕೋಟಿಯ (not ಕೋಟಿದ)
        # ============================================================
        suffix_crore_standalone = pynutil.insert(" ಕೋಟಿ")
        suffix_crore_genitive = pynutil.insert(" ಕೋಟಿಯ")

        # Standalone crores
        graph_crores_standalone = digit + (delete_zero**7) + suffix_crore_standalone
        graph_ten_crores_standalone = teens_and_ties + (delete_zero**7) + suffix_crore_standalone

        # Crores with remainder
        graph_crores_with_digit = digit + (delete_zero**6) + suffix_crore_genitive + insert_space + digit
        graph_crores_with_teens = digit + (delete_zero**5) + suffix_crore_genitive + insert_space + teens_ties
        graph_crores_with_hundreds = digit + (delete_zero**4) + suffix_crore_genitive + insert_space + graph_hundreds
        graph_crores_with_thousands = digit + (delete_zero**3) + suffix_crore_genitive + insert_space + graph_thousands
        graph_crores_with_ten_thousands = digit + (delete_zero**2) + suffix_crore_genitive + insert_space + graph_ten_thousands
        graph_crores_with_lakhs = digit + delete_zero + suffix_crore_genitive + insert_space + graph_lakhs
        graph_crores_with_ten_lakhs = digit + suffix_crore_genitive + insert_space + graph_ten_lakhs

        graph_ten_crores_with_digit = teens_and_ties + (delete_zero**6) + suffix_crore_genitive + insert_space + digit
        graph_ten_crores_with_teens = teens_and_ties + (delete_zero**5) + suffix_crore_genitive + insert_space + teens_ties
        graph_ten_crores_with_hundreds = teens_and_ties + (delete_zero**4) + suffix_crore_genitive + insert_space + graph_hundreds
        graph_ten_crores_with_thousands = teens_and_ties + (delete_zero**3) + suffix_crore_genitive + insert_space + graph_thousands
        graph_ten_crores_with_ten_thousands = teens_and_ties + (delete_zero**2) + suffix_crore_genitive + insert_space + graph_ten_thousands
        graph_ten_crores_with_lakhs = teens_and_ties + delete_zero + suffix_crore_genitive + insert_space + graph_lakhs
        graph_ten_crores_with_ten_lakhs = teens_and_ties + suffix_crore_genitive + insert_space + graph_ten_lakhs

        graph_crores = (
            graph_crores_standalone
            | graph_crores_with_digit
            | graph_crores_with_teens
            | graph_crores_with_hundreds
            | graph_crores_with_thousands
            | graph_crores_with_ten_thousands
            | graph_crores_with_lakhs
            | graph_crores_with_ten_lakhs
        )
        graph_crores.optimize()
        self.graph_crores = graph_crores

        graph_ten_crores = (
            graph_ten_crores_standalone
            | graph_ten_crores_with_digit
            | graph_ten_crores_with_teens
            | graph_ten_crores_with_hundreds
            | graph_ten_crores_with_thousands
            | graph_ten_crores_with_ten_thousands
            | graph_ten_crores_with_lakhs
            | graph_ten_crores_with_ten_lakhs
        )
        graph_ten_crores.optimize()
        self.graph_ten_crores = graph_ten_crores

        # ============================================================
        # HUNDRED CRORES and above (10^9+)
        # Instead of ಅರಬ್/ಖರಬ್, use natural Kannada: ನೂರು ಕೋಟಿ, ಸಾವಿರ ಕೋಟಿ
        # ============================================================
        
        # Combined graph for any crore remainder (1 to 99,99,99,999)
        graph_crore_remainder = (
            digit
            | teens_and_ties
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
        )
        
        # Hundred crores (10^9): 100-999 ಕೋಟಿ with any remainder
        graph_hundred_crores_standalone = graph_hundreds + (delete_zero**7) + suffix_crore_standalone
        graph_hundred_crores_with_digit = graph_hundreds + (delete_zero**6) + suffix_crore_genitive + insert_space + digit
        graph_hundred_crores_with_teens = graph_hundreds + (delete_zero**5) + suffix_crore_genitive + insert_space + teens_ties
        graph_hundred_crores_with_hundreds = graph_hundreds + (delete_zero**4) + suffix_crore_genitive + insert_space + graph_hundreds
        graph_hundred_crores_with_thousands = graph_hundreds + (delete_zero**3) + suffix_crore_genitive + insert_space + graph_thousands
        graph_hundred_crores_with_ten_thousands = graph_hundreds + (delete_zero**2) + suffix_crore_genitive + insert_space + graph_ten_thousands
        graph_hundred_crores_with_lakhs = graph_hundreds + delete_zero + suffix_crore_genitive + insert_space + graph_lakhs
        graph_hundred_crores_with_ten_lakhs = graph_hundreds + suffix_crore_genitive + insert_space + graph_ten_lakhs
        
        graph_hundred_crores = (
            graph_hundred_crores_standalone
            | graph_hundred_crores_with_digit
            | graph_hundred_crores_with_teens
            | graph_hundred_crores_with_hundreds
            | graph_hundred_crores_with_thousands
            | graph_hundred_crores_with_ten_thousands
            | graph_hundred_crores_with_lakhs
            | graph_hundred_crores_with_ten_lakhs
        )
        graph_hundred_crores.optimize()
        
        # Thousand crores (10^10): 1000-9999 ಕೋಟಿ
        # Use coefficient patterns that output ಸಾವಿರದ (not ಸಾವಿರ ಕೋಟಿ) and append ಕೋಟಿ at end
        # This prevents ಕೋಟಿ from appearing twice
        
        # For X000 crore (e.g., 1000 crore = 10000000000)
        graph_thousand_crores_standalone = digit + (delete_zero**10) + pynutil.insert(" ಸಾವಿರ ಕೋಟಿ")
        
        # For X000 crore with sub-crore remainder (uses "ಸಾವಿರ ಕೋಟಿಯ")
        suffix_thousand_crore_genitive = pynutil.insert(" ಸಾವಿರ ಕೋಟಿಯ")
        graph_thousand_crores_with_digit = digit + (delete_zero**9) + suffix_thousand_crore_genitive + insert_space + digit
        graph_thousand_crores_with_teens = digit + (delete_zero**8) + suffix_thousand_crore_genitive + insert_space + teens_ties
        graph_thousand_crores_with_hundreds = digit + (delete_zero**7) + suffix_thousand_crore_genitive + insert_space + graph_hundreds
        graph_thousand_crores_with_thousands = digit + (delete_zero**6) + suffix_thousand_crore_genitive + insert_space + graph_thousands
        graph_thousand_crores_with_ten_thousands = digit + (delete_zero**5) + suffix_thousand_crore_genitive + insert_space + graph_ten_thousands
        graph_thousand_crores_with_lakhs = digit + (delete_zero**4) + suffix_thousand_crore_genitive + insert_space + graph_lakhs
        graph_thousand_crores_with_ten_lakhs = digit + (delete_zero**3) + suffix_thousand_crore_genitive + insert_space + graph_ten_lakhs
        
        # For X00Y, X0YZ, XYZW crore patterns (e.g., 1001 crore, 1234 crore)
        # Use "ಸಾವಿರದ" (genitive), not "ಸಾವಿರ ಕೋಟಿ", then append ಕೋಟಿ at end
        # X00Y crore standalone (e.g., 1001 crore = 10010000000)
        graph_thousand_crores_x00y = (digit + (delete_zero**2) + suffix_thousand_genitive + insert_space + digit 
                                      + (delete_zero**7) + suffix_crore_standalone)
        # X0YZ crore standalone (e.g., 1023 crore)
        graph_thousand_crores_x0yz = (digit + delete_zero + suffix_thousand_genitive + insert_space + teens_ties
                                      + (delete_zero**7) + suffix_crore_standalone)
        # XYZW crore standalone (e.g., 1234 crore)
        graph_thousand_crores_xyzw = (digit + suffix_thousand_genitive + insert_space + graph_hundreds
                                      + (delete_zero**7) + suffix_crore_standalone)
        
        # X00Y, X0YZ, XYZW crore with sub-crore remainder
        graph_thousand_crores_x00y_with_rem = (digit + (delete_zero**2) + suffix_thousand_genitive + insert_space + digit
                                               + suffix_crore_genitive + insert_space + graph_ten_lakhs)
        graph_thousand_crores_x0yz_with_rem = (digit + delete_zero + suffix_thousand_genitive + insert_space + teens_ties
                                               + suffix_crore_genitive + insert_space + graph_ten_lakhs)
        graph_thousand_crores_xyzw_with_rem = (digit + suffix_thousand_genitive + insert_space + graph_hundreds
                                               + suffix_crore_genitive + insert_space + graph_ten_lakhs)
        
        graph_thousand_crores = (
            graph_thousand_crores_standalone
            | graph_thousand_crores_with_digit
            | graph_thousand_crores_with_teens
            | graph_thousand_crores_with_hundreds
            | graph_thousand_crores_with_thousands
            | graph_thousand_crores_with_ten_thousands
            | graph_thousand_crores_with_lakhs
            | graph_thousand_crores_with_ten_lakhs
            | graph_thousand_crores_x00y
            | graph_thousand_crores_x0yz
            | graph_thousand_crores_xyzw
            | graph_thousand_crores_x00y_with_rem
            | graph_thousand_crores_x0yz_with_rem
            | graph_thousand_crores_xyzw_with_rem
        )
        graph_thousand_crores.optimize()
        
        # Ten thousand crores (10^11): 10000-99999 ಕೋಟಿ
        # Use coefficient patterns with ಸಾವಿರದ and append ಕೋಟಿ at end
        
        # For XY000 crore (e.g., 12000 crore = 120000000000)
        graph_ten_thousand_crores_standalone = teens_and_ties + (delete_zero**10) + pynutil.insert(" ಸಾವಿರ ಕೋಟಿ")
        
        # For XY000 crore with sub-crore remainder
        suffix_ten_thousand_crore_genitive = pynutil.insert(" ಸಾವಿರ ಕೋಟಿಯ")
        graph_ten_thousand_crores_with_digit = teens_and_ties + (delete_zero**9) + suffix_ten_thousand_crore_genitive + insert_space + digit
        graph_ten_thousand_crores_with_teens = teens_and_ties + (delete_zero**8) + suffix_ten_thousand_crore_genitive + insert_space + teens_ties
        graph_ten_thousand_crores_with_hundreds = teens_and_ties + (delete_zero**7) + suffix_ten_thousand_crore_genitive + insert_space + graph_hundreds
        graph_ten_thousand_crores_with_thousands = teens_and_ties + (delete_zero**6) + suffix_ten_thousand_crore_genitive + insert_space + graph_thousands
        graph_ten_thousand_crores_with_ten_thousands = teens_and_ties + (delete_zero**5) + suffix_ten_thousand_crore_genitive + insert_space + graph_ten_thousands
        graph_ten_thousand_crores_with_lakhs = teens_and_ties + (delete_zero**4) + suffix_ten_thousand_crore_genitive + insert_space + graph_lakhs
        graph_ten_thousand_crores_with_ten_lakhs = teens_and_ties + (delete_zero**3) + suffix_ten_thousand_crore_genitive + insert_space + graph_ten_lakhs
        
        # For XY00Z, XY0ZW, XYZWV crore patterns (e.g., 12001, 12034, 12345 crore)
        # Use "ಸಾವಿರದ" (genitive), not "ಸಾವಿರ ಕೋಟಿ", then append ಕೋಟಿ at end
        # XY00Z crore standalone
        graph_ten_thousand_crores_xy00z = (teens_and_ties + (delete_zero**2) + suffix_thousand_genitive + insert_space + digit
                                           + (delete_zero**7) + suffix_crore_standalone)
        # XY0ZW crore standalone
        graph_ten_thousand_crores_xy0zw = (teens_and_ties + delete_zero + suffix_thousand_genitive + insert_space + teens_ties
                                           + (delete_zero**7) + suffix_crore_standalone)
        # XYZWV crore standalone (e.g., 12345 crore)
        graph_ten_thousand_crores_xyzwv = (teens_and_ties + suffix_thousand_genitive + insert_space + graph_hundreds
                                           + (delete_zero**7) + suffix_crore_standalone)
        
        # With sub-crore remainder
        graph_ten_thousand_crores_xy00z_with_rem = (teens_and_ties + (delete_zero**2) + suffix_thousand_genitive + insert_space + digit
                                                    + suffix_crore_genitive + insert_space + graph_ten_lakhs)
        graph_ten_thousand_crores_xy0zw_with_rem = (teens_and_ties + delete_zero + suffix_thousand_genitive + insert_space + teens_ties
                                                    + suffix_crore_genitive + insert_space + graph_ten_lakhs)
        graph_ten_thousand_crores_xyzwv_with_rem = (teens_and_ties + suffix_thousand_genitive + insert_space + graph_hundreds
                                                    + suffix_crore_genitive + insert_space + graph_ten_lakhs)
        
        graph_ten_thousand_crores = (
            graph_ten_thousand_crores_standalone
            | graph_ten_thousand_crores_with_digit
            | graph_ten_thousand_crores_with_teens
            | graph_ten_thousand_crores_with_hundreds
            | graph_ten_thousand_crores_with_thousands
            | graph_ten_thousand_crores_with_ten_thousands
            | graph_ten_thousand_crores_with_lakhs
            | graph_ten_thousand_crores_with_ten_lakhs
            | graph_ten_thousand_crores_xy00z
            | graph_ten_thousand_crores_xy0zw
            | graph_ten_thousand_crores_xyzwv
            | graph_ten_thousand_crores_xy00z_with_rem
            | graph_ten_thousand_crores_xy0zw_with_rem
            | graph_ten_thousand_crores_xyzwv_with_rem
        )
        graph_ten_thousand_crores.optimize()

        # Lakh crores (10^12): 1-9 lakh ಕೋಟಿ
        # Use "ಲಕ್ಷದ" (genitive) for coefficient composition, append ಕೋಟಿ at end
        
        # For X00000 crore (e.g., 1 lakh crore = 1000000000000)
        graph_lakh_crores_standalone = digit + (delete_zero**12) + pynutil.insert(" ಲಕ್ಷ ಕೋಟಿ")
        
        # For X00000 crore with sub-crore remainder (uses "ಲಕ್ಷ ಕೋಟಿಯ")
        suffix_lakh_crore_genitive = pynutil.insert(" ಲಕ್ಷ ಕೋಟಿಯ")
        graph_lakh_crores_with_digit = digit + (delete_zero**11) + suffix_lakh_crore_genitive + insert_space + digit
        graph_lakh_crores_with_teens = digit + (delete_zero**10) + suffix_lakh_crore_genitive + insert_space + teens_ties
        graph_lakh_crores_with_hundreds = digit + (delete_zero**9) + suffix_lakh_crore_genitive + insert_space + graph_hundreds
        graph_lakh_crores_with_thousands = digit + (delete_zero**8) + suffix_lakh_crore_genitive + insert_space + graph_thousands
        graph_lakh_crores_with_ten_thousands = digit + (delete_zero**7) + suffix_lakh_crore_genitive + insert_space + graph_ten_thousands
        graph_lakh_crores_with_lakhs = digit + (delete_zero**6) + suffix_lakh_crore_genitive + insert_space + graph_lakhs
        graph_lakh_crores_with_ten_lakhs = digit + (delete_zero**5) + suffix_lakh_crore_genitive + insert_space + graph_ten_lakhs
        
        # For X000YZ, X00YZW, etc. crore patterns (coefficient > 1 lakh)
        # Use "ಲಕ್ಷದ" for coefficient, append ಕೋಟಿ at end (avoid crore duplication)
        # X000Y crore (e.g., 100100 crore = 1,00,100 crore = 1 lakh 100 crore)
        graph_lakh_crores_x000y = (digit + (delete_zero**4) + suffix_lakh_genitive + insert_space + digit
                                  + (delete_zero**7) + suffix_crore_standalone)
        graph_lakh_crores_x00yz = (digit + (delete_zero**3) + suffix_lakh_genitive + insert_space + teens_ties
                                  + (delete_zero**7) + suffix_crore_standalone)
        graph_lakh_crores_x0yzw = (digit + (delete_zero**2) + suffix_lakh_genitive + insert_space + graph_hundreds
                                  + (delete_zero**7) + suffix_crore_standalone)
        graph_lakh_crores_xyzwv = (digit + delete_zero + suffix_lakh_genitive + insert_space + graph_thousands
                                  + (delete_zero**7) + suffix_crore_standalone)
        graph_lakh_crores_full = (digit + suffix_lakh_genitive + insert_space + graph_ten_thousands
                                  + (delete_zero**7) + suffix_crore_standalone)
        
        graph_lakh_crores = (
            graph_lakh_crores_standalone
            | graph_lakh_crores_with_digit
            | graph_lakh_crores_with_teens
            | graph_lakh_crores_with_hundreds
            | graph_lakh_crores_with_thousands
            | graph_lakh_crores_with_ten_thousands
            | graph_lakh_crores_with_lakhs
            | graph_lakh_crores_with_ten_lakhs
            | graph_lakh_crores_x000y
            | graph_lakh_crores_x00yz
            | graph_lakh_crores_x0yzw
            | graph_lakh_crores_xyzwv
            | graph_lakh_crores_full
        )
        graph_lakh_crores.optimize()

        # Ten lakh crores (10^13): 10-99 lakh ಕೋಟಿ
        # Use "ಲಕ್ಷದ" for coefficient composition, append ಕೋಟಿ at end
        
        # For XY00000 crore (e.g., 10 lakh crore)
        graph_ten_lakh_crores_standalone = teens_and_ties + (delete_zero**12) + pynutil.insert(" ಲಕ್ಷ ಕೋಟಿ")
        
        # With sub-crore remainder
        suffix_ten_lakh_crore_genitive = pynutil.insert(" ಲಕ್ಷ ಕೋಟಿಯ")
        graph_ten_lakh_crores_with_digit = teens_and_ties + (delete_zero**11) + suffix_ten_lakh_crore_genitive + insert_space + digit
        graph_ten_lakh_crores_with_teens = teens_and_ties + (delete_zero**10) + suffix_ten_lakh_crore_genitive + insert_space + teens_ties
        graph_ten_lakh_crores_with_hundreds = teens_and_ties + (delete_zero**9) + suffix_ten_lakh_crore_genitive + insert_space + graph_hundreds
        graph_ten_lakh_crores_with_thousands = teens_and_ties + (delete_zero**8) + suffix_ten_lakh_crore_genitive + insert_space + graph_thousands
        graph_ten_lakh_crores_with_ten_thousands = teens_and_ties + (delete_zero**7) + suffix_ten_lakh_crore_genitive + insert_space + graph_ten_thousands
        graph_ten_lakh_crores_with_lakhs = teens_and_ties + (delete_zero**6) + suffix_ten_lakh_crore_genitive + insert_space + graph_lakhs
        graph_ten_lakh_crores_with_ten_lakhs = teens_and_ties + (delete_zero**5) + suffix_ten_lakh_crore_genitive + insert_space + graph_ten_lakhs
        
        # For XY000Z, XY00ZW, etc. crore patterns
        graph_ten_lakh_crores_xy000z = (teens_and_ties + (delete_zero**4) + suffix_lakh_genitive + insert_space + digit
                                        + (delete_zero**7) + suffix_crore_standalone)
        graph_ten_lakh_crores_xy00zw = (teens_and_ties + (delete_zero**3) + suffix_lakh_genitive + insert_space + teens_ties
                                        + (delete_zero**7) + suffix_crore_standalone)
        graph_ten_lakh_crores_xy0zwv = (teens_and_ties + (delete_zero**2) + suffix_lakh_genitive + insert_space + graph_hundreds
                                        + (delete_zero**7) + suffix_crore_standalone)
        graph_ten_lakh_crores_xyzwvu = (teens_and_ties + delete_zero + suffix_lakh_genitive + insert_space + graph_thousands
                                        + (delete_zero**7) + suffix_crore_standalone)
        graph_ten_lakh_crores_full = (teens_and_ties + suffix_lakh_genitive + insert_space + graph_ten_thousands
                                      + (delete_zero**7) + suffix_crore_standalone)
        
        graph_ten_lakh_crores = (
            graph_ten_lakh_crores_standalone
            | graph_ten_lakh_crores_with_digit
            | graph_ten_lakh_crores_with_teens
            | graph_ten_lakh_crores_with_hundreds
            | graph_ten_lakh_crores_with_thousands
            | graph_ten_lakh_crores_with_ten_thousands
            | graph_ten_lakh_crores_with_lakhs
            | graph_ten_lakh_crores_with_ten_lakhs
            | graph_ten_lakh_crores_xy000z
            | graph_ten_lakh_crores_xy00zw
            | graph_ten_lakh_crores_xy0zwv
            | graph_ten_lakh_crores_xyzwvu
            | graph_ten_lakh_crores_full
        )
        graph_ten_lakh_crores.optimize()

        # ============================================================
        # FINAL GRAPH COMPOSITION
        # ============================================================
        graph_without_leading_zeros = (
            digit
            | zero
            | teens_and_ties
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | graph_hundred_crores
            | graph_thousand_crores
            | graph_ten_thousand_crores
            | graph_lakh_crores
            | graph_ten_lakh_crores
        )
        self.graph_without_leading_zeros = graph_without_leading_zeros.optimize()

        # Handle numbers with leading zeros by reading digit-by-digit
        cardinal_with_leading_zeros = pynini.compose(
            NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph
        )
        cardinal_with_leading_zeros = pynutil.add_weight(cardinal_with_leading_zeros, 0.5)

        # Full graph including leading zeros (without commas)
        graph_no_commas = graph_without_leading_zeros | cardinal_with_leading_zeros

        # ============================================================
        # COMMA-SEPARATED NUMBERS (strict validation)
        # ============================================================
        delete_comma = pynutil.delete(",")
        
        def exactly_n_digits(n):
            return pynini.closure(NEMO_ALL_DIGIT, n, n)
        
        # Western format: 1,000 | 1,000,000
        western_format = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 3) +
            pynini.closure(delete_comma + exactly_n_digits(3), 1)
        )
        
        # Indian format: 1,00,000 | 12,34,567
        indian_format = (
            pynini.closure(NEMO_ALL_DIGIT, 1, 2) +
            pynini.closure(delete_comma + exactly_n_digits(2)) +
            delete_comma + exactly_n_digits(3)
        )
        
        comma_number = western_format | indian_format
        cardinal_with_commas = pynini.compose(comma_number, graph_without_leading_zeros)
        cardinal_with_commas = pynutil.add_weight(cardinal_with_commas, 0.1)

        # Final graph
        final_graph = graph_no_commas | cardinal_with_commas

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph