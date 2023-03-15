# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    NEMO_ALNUM,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_LOWER_NOT_A,
    GraphFst,
    capitalized_input_graph,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


def get_serial_number(cardinal):
    """
    any alphanumerical character sequence with at least one number with length greater equal to 3 and
    excluding any numeric sequence containing double digits (ties/teens) preceded by 'a'.
    This avoids cases like "a thirty six" being converted to "a36"  in "a thirty six times increase"
    """

    digit = pynini.compose(cardinal.graph_no_exception, NEMO_DIGIT)
    two_digit = pynutil.add_weight(pynini.compose(cardinal.graph_two_digit, NEMO_DIGIT ** 2), 0.002)
    character = digit | two_digit | NEMO_ALPHA
    sequence = (NEMO_LOWER_NOT_A | digit) + pynini.closure(pynutil.delete(" ") + character, 2)
    sequence |= character + pynini.closure(pynutil.delete(" ") + (digit | NEMO_ALPHA), 2)
    sequence2 = (
        NEMO_ALPHA
        + pynini.closure(pynutil.delete(" ") + NEMO_ALPHA, 1)
        + pynini.closure(pynutil.delete(" ") + two_digit, 1)
    )
    sequence2 |= NEMO_LOWER_NOT_A + pynini.closure(pynutil.delete(" ") + two_digit, 1)
    sequence2 |= (
        two_digit
        + pynini.closure(pynutil.delete(" ") + two_digit, 1)
        + pynini.closure(pynutil.delete(" ") + NEMO_ALPHA, 1)
    )
    sequence = (sequence | sequence2) @ (pynini.closure(NEMO_ALNUM) + NEMO_DIGIT + pynini.closure(NEMO_ALNUM))
    return sequence.optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g. 
        one two three one two three five six seven eight -> { number_part: "123-123-5678" }

    This class also support card number and IP format.
        "one two three dot one double three dot o dot four o" -> { number_part: "123.133.0.40"}

        "three two double seven three two one four three two one four three double zero five" ->
            { number_part: 3277 3214 3214 3005}

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="telephone", kind="classify")
        # country code, number_part, extension
        digit_to_str = (
            pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize())
            | pynini.cross("0", pynini.union("o", "oh", "zero")).optimize()
        )

        str_to_digit = pynini.invert(digit_to_str)
        if input_case == INPUT_CASED:
            str_to_digit = capitalized_input_graph(str_to_digit)

        double_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_to_str, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_to_str, "output"),
                    pynutil.insert("double ") + pynini.project(str(i) @ digit_to_str, "output"),
                )
                for i in range(10)
            ]
        )
        double_digit.invert()

        triple_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_to_str, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_to_str, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_to_str, "output"),
                    pynutil.insert("triple ") + pynini.project(str(i) @ digit_to_str, "output"),
                )
                for i in range(10)
            ]
        )
        triple_digit.invert()

        # to handle cases like "one twenty three"
        two_digit_cardinal = pynini.compose(cardinal.graph_no_exception, NEMO_DIGIT ** 2)
        double_digit_to_digit = (
            pynini.compose(double_digit, str_to_digit + pynutil.delete(" ") + str_to_digit) | two_digit_cardinal
        )
        triple_digit_to_digit = pynini.compose(
            triple_digit, str_to_digit + delete_space + str_to_digit + delete_space + str_to_digit
        )
        single_or_double_digit = (pynutil.add_weight(double_digit_to_digit, -0.0001) | str_to_digit).optimize()
        single_double_or_triple_digit = (
            pynutil.add_weight(triple_digit_to_digit, -0.0001) | single_or_double_digit | delete_space
        ).optimize()

        single_or_double_digit |= (
            single_or_double_digit
            + pynini.closure(pynutil.add_weight(pynutil.delete(" ") + single_or_double_digit, 0.0001))
        ).optimize()
        single_double_or_triple_digit |= (
            single_double_or_triple_digit
            + pynini.closure(pynutil.add_weight(pynutil.delete(" ") + single_double_or_triple_digit, 0.0001))
        ).optimize()

        number_part = pynini.compose(
            single_double_or_triple_digit,
            NEMO_DIGIT ** 3 + pynutil.insert("-") + NEMO_DIGIT ** 3 + pynutil.insert("-") + NEMO_DIGIT ** 4,
        ).optimize()
        number_part = pynutil.insert("number_part: \"") + number_part.optimize() + pynutil.insert("\"")

        cardinal_option = pynini.compose(single_double_or_triple_digit, NEMO_DIGIT ** (2, 3))

        country_code = (
            pynutil.insert("country_code: \"")
            + pynini.closure(pynini.cross("plus ", "+"), 0, 1)
            + ((pynini.closure(str_to_digit + pynutil.delete(" "), 0, 2) + str_to_digit) | cardinal_option)
            + pynutil.insert("\"")
        )

        optional_country_code = pynini.closure(country_code + pynutil.delete(" ") + insert_space, 0, 1).optimize()
        graph = optional_country_code + number_part

        # credit card number
        space_four_digits = insert_space + NEMO_DIGIT ** 4
        space_five_digits = space_four_digits + NEMO_DIGIT
        space_six_digits = space_five_digits + NEMO_DIGIT
        credit_card_graph = pynini.compose(
            single_double_or_triple_digit,
            NEMO_DIGIT ** 4 + (space_six_digits | (space_four_digits ** 2)) + space_four_digits,
        ).optimize()

        credit_card_graph |= pynini.compose(
            single_double_or_triple_digit, NEMO_DIGIT ** 4 + space_six_digits + space_five_digits
        ).optimize()

        graph |= pynutil.insert("number_part: \"") + credit_card_graph.optimize() + pynutil.insert("\"")

        # SSN
        ssn_graph = pynini.compose(
            single_double_or_triple_digit,
            NEMO_DIGIT ** 3 + pynutil.insert("-") + NEMO_DIGIT ** 2 + pynutil.insert("-") + NEMO_DIGIT ** 4,
        ).optimize()
        graph |= pynutil.insert("number_part: \"") + ssn_graph.optimize() + pynutil.insert("\"")

        # ip
        digit_or_double = pynini.closure(str_to_digit + pynutil.delete(" "), 0, 1) + double_digit_to_digit
        digit_or_double |= double_digit_to_digit + pynini.closure(pynutil.delete(" ") + str_to_digit, 0, 1)
        digit_or_double |= str_to_digit + (pynutil.delete(" ") + str_to_digit) ** (0, 2)
        digit_or_double |= cardinal_option
        digit_or_double = digit_or_double.optimize()

        ip_graph = digit_or_double + (pynini.cross(" dot ", ".") + digit_or_double) ** 3

        graph |= (
            pynutil.insert("number_part: \"")
            + pynutil.add_weight(ip_graph.optimize(), MIN_NEG_WEIGHT)
            + pynutil.insert("\"")
        )

        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph)

        # serial graph shouldn't apply TO_LOWER
        graph |= (
            pynutil.insert("number_part: \"")
            + pynutil.add_weight(get_serial_number(cardinal=cardinal), weight=0.0001)
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
