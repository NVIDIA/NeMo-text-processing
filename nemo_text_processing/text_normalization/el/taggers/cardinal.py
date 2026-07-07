# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.el.graph_utils import shift_cardinal_gender_fem
from nemo_text_processing.text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, insert_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals in Greek, e.g.
        55 -> cardinal { integer: "πενήντα πέντε" }
        123 -> cardinal { integer: "εκατόν είκοσι τρία" }

    Numbers are rendered in the neuter citation form. Supports 0 up to 999999999999
    (i.e. up to hundreds of billions). Greek numerals inflect for gender; the thousands
    multiplier is feminine ("τρεις χιλιάδες"), while the units block and the
    million/billion multipliers stay neuter.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))  # 1-9 -> word
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))  # 10-19 -> word
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))  # 2-9 -> tens word
        graph_hundreds_map = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))  # 2-9 -> hundreds word

        graph_digit_no_one = (NEMO_DIGIT - "1") @ graph_digit

        self.zero = pynini.cross("0", "μηδέν")

        # single_digits_graph: maps digit characters → Greek words with spaces between
        # e.g., "123" → "ένα δύο τρία"
        single_digits_graph = graph_digit | self.zero
        self.single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)

        # --- tens: 10-99 ---
        tens_exact = graph_ties + pynutil.delete("0")  # 20, 30, ... 90
        tens_unit = graph_ties + insert_space + graph_digit  # 21-99 with non-zero unit
        graph_tens = graph_teen | tens_exact | tens_unit
        self.graph_tens = graph_tens.optimize()

        # remainder for the last two digits of a hundreds group (01-99, non-zero)
        two_digit_non_zero = (pynutil.delete("0") + insert_space + graph_digit) | (insert_space + graph_tens)

        # --- hundreds: 100-999 ---
        hundred_one_exact = pynini.cross("100", "εκατό")
        hundred_one_rem = pynini.cross("1", "εκατόν") + two_digit_non_zero  # 101-199
        hundreds_exact = graph_hundreds_map + pynutil.delete("00")  # 200, 300, ... 900
        hundreds_rem = graph_hundreds_map + two_digit_non_zero  # 201-999
        graph_hundreds = hundred_one_exact | hundred_one_rem | hundreds_exact | hundreds_rem
        self.graph_hundreds = graph_hundreds.optimize()

        # numbers 1-999 (neuter, variable length, no leading zeros)
        graph_one_to_999 = graph_digit | graph_tens | graph_hundreds
        self.graph_one_to_999 = graph_one_to_999.optimize()

        # multiplier for the thousands group (2-999); the standalone "1" is handled as "χίλια"
        multiplier_base = graph_digit_no_one | graph_tens | graph_hundreds
        fem_multiplier = shift_cardinal_gender_fem(multiplier_base)

        # --- thousands: 1000-999999 (variable length, no leading zeros) ---
        thousand_one = pynini.cross("1", "χίλια")
        thousands_head = thousand_one | (fem_multiplier + insert_space + pynutil.insert("χιλιάδες"))
        rem3 = (
            pynutil.delete("000")
            | (pynutil.delete("00") + insert_space + graph_digit)
            | (pynutil.delete("0") + insert_space + graph_tens)
            | (insert_space + graph_hundreds)
        )
        graph_thousands = thousands_head + rem3
        self.graph_thousands = graph_thousands.optimize()

        # --- fixed-width 3-digit group blocks, used for the remainders of large numbers ---
        # a non-zero group of exactly three digits (001-999), neuter
        group_nonzero = (
            (pynutil.delete("00") + graph_digit)
            | (pynutil.delete("0") + graph_tens)
            | graph_hundreds
        )
        # same but excluding 001 (used where "one" needs a scale word instead)
        group_nonzero_no_one = (
            (pynutil.delete("00") + graph_digit_no_one)
            | (pynutil.delete("0") + graph_tens)
            | graph_hundreds
        )

        # each *_group_sp consumes exactly three digits and emits a leading space when non-zero
        units_group_sp = pynutil.delete("000") | (insert_space + group_nonzero)
        thousands_group_sp = (
            pynutil.delete("000")
            | (insert_space + pynini.cross("001", "χίλια"))
            | (insert_space + shift_cardinal_gender_fem(group_nonzero_no_one) + insert_space + pynutil.insert("χιλιάδες"))
        )
        millions_group_sp = (
            pynutil.delete("000")
            | (insert_space + pynini.cross("001", "ένα εκατομμύριο"))
            | (insert_space + group_nonzero_no_one + insert_space + pynutil.insert("εκατομμύρια"))
        )

        # --- millions: 1000000 - 999999999 (7-9 digits) ---
        million_one = pynini.cross("1", "ένα") + insert_space + pynutil.insert("εκατομμύριο")
        million_multi = multiplier_base + insert_space + pynutil.insert("εκατομμύρια")
        million_head = million_one | million_multi
        rem6 = thousands_group_sp + units_group_sp
        graph_millions = million_head + rem6
        self.graph_millions = graph_millions.optimize()

        # --- billions: 1000000000 - 999999999999 (10-12 digits) ---
        billion_one = pynini.cross("1", "ένα") + insert_space + pynutil.insert("δισεκατομμύριο")
        billion_multi = multiplier_base + insert_space + pynutil.insert("δισεκατομμύρια")
        billion_head = billion_one | billion_multi
        rem9 = millions_group_sp + thousands_group_sp + units_group_sp
        graph_billions = billion_head + rem9
        self.graph_billions = graph_billions.optimize()

        graph = (
            self.zero
            | graph_one_to_999
            | graph_thousands
            | graph_millions
            | graph_billions
        )
        self.graph_no_tokens = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph_no_tokens + pynutil.insert("\"")
        self.final_graph = final_graph
        self.fst = self.add_tokens(final_graph).optimize()
