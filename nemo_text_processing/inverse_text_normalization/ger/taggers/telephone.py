# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    NEMO_SPACE,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone.
    The grammar assumes telephone number formatting used in Germany and Austria (DIN 5008).
    For more info go to: https://de.wikipedia.org/wiki/DIN_5008
        e.g. null eins fünf zwei fünf sechs sieben drei vier zwei eins -> telephone { country_code: "015" number_part: "25 673421" }
        e.g. zwei vier fünf eins vier vierzehn zehn -> telephone { country_code: "245" number_part: "141410" }
        e.g. null fünf fünf eins vier acht null null acht -> telephone { number_part: "055 148008" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")
        graph_zero = pynini.cross("null", "0")
        graph_single_digits = cardinal.digits
        graph_double_digits = cardinal.graph_double_digits
        graph_cardinal = cardinal.graph_all_cardinals

        # Handles country codes of the following formats: 00#, 0##, ###
        # The block below is also suitable to handle area codes
        country_code_three_digit = (
            (
                pynini.closure((graph_zero + pynutil.delete(NEMO_SPACE)), 2, 2)
                + graph_single_digits
            )
            | (
                (graph_zero + pynutil.delete(NEMO_SPACE))
                + (
                    (
                        graph_single_digits
                        + pynutil.delete(NEMO_SPACE)
                        + (graph_single_digits | graph_zero)
                    )
                    | graph_double_digits
                )
            )
            | (
                (
                    graph_single_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    graph_single_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | graph_cardinal
            )
        )

        # Handles the country codes including the "+" sign
        # The following formats are accepted: +#, +##, +###
        country_code_plus = (
            pynini.cross("plus", "+")
            + pynutil.delete(NEMO_SPACE)
            + (
                (graph_single_digits)
                | (
                    (
                        graph_single_digits
                        + pynutil.delete(NEMO_SPACE)
                        + (graph_single_digits | graph_zero)
                    )
                    | graph_double_digits
                )
                | (
                    (
                        graph_single_digits
                        + pynutil.delete(NEMO_SPACE)
                        + (graph_single_digits | graph_zero)
                        + pynutil.delete(NEMO_SPACE)
                        + (graph_single_digits | graph_zero)
                    )
                    | (
                        graph_single_digits
                        + pynutil.delete(NEMO_SPACE)
                        + graph_double_digits
                    )
                    | (
                        graph_double_digits
                        + pynutil.delete(NEMO_SPACE)
                        + (graph_single_digits | graph_zero)
                    )
                    | graph_cardinal
                )
            )
        )

        graph_country_code = (
            pynutil.insert("country_code:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + (country_code_three_digit | country_code_plus)
            + pynutil.insert('"')
        )

        # Handles two-digit area codes
        # Covers the following formats: 0#, ##
        two_digit_code = (
            (graph_zero + pynutil.delete(NEMO_SPACE) + graph_single_digits)
            | (
                graph_single_digits
                + pynutil.delete(NEMO_SPACE)
                + (graph_single_digits | graph_zero)
            )
            | graph_double_digits
        )

        # Handles three-digit area codes w/o the leading zeros (###)
        three_digit_code = (
            (
                graph_single_digits
                + pynutil.delete(NEMO_SPACE)
                + graph_single_digits
                + pynutil.delete(NEMO_SPACE)
                + graph_single_digits
            )
            | (graph_single_digits + pynutil.delete(NEMO_SPACE) + graph_double_digits)
            | (graph_double_digits + pynutil.delete(NEMO_SPACE) + graph_single_digits)
            | graph_cardinal
        )

        # There are 24 different ways in which a six-digit number can be conveyed with 1, 2, and 3-dgit chunks
        # These can be expressed by a sequence of 2-6 transducers
        # Chunks longer than 3 digits (e.g. one thousand two hundred thirty four) are uncommon in everyday speech and thus excluded from the grammar
        number_part = (
            (
                (
                    country_code_three_digit
                    | two_digit_code
                    | three_digit_code
                    | graph_single_digits
                )
                + pynini.accep(NEMO_SPACE)
            ).ques
            + (
                pynutil.add_weight(
                    (graph_cardinal + pynutil.delete(NEMO_SPACE) + graph_cardinal),
                    0.01,
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    graph_cardinal
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_double_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    graph_double_digits
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
            )
        ) | (
            three_digit_code
        )  # This final acceptor handles emergency and public service numbers

        graph_number_part = (
            pynutil.insert("number_part:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + number_part
            + pynutil.insert('"')
        )

        # Handles extensions
        # Exceptions are in the format: "-##" or "-#"
        durchwahl = pynini.accep("Durchwahl") | pynini.accep("durchwahl")
        extension = (
            pynini.cross(durchwahl, "-")
            + pynutil.delete(NEMO_SPACE)
            + (
                (
                    (graph_single_digits | graph_zero)
                    + pynutil.delete(NEMO_SPACE)
                    + (graph_single_digits | graph_zero)
                )
                | graph_double_digits
            )
        )

        graph_extension = (
            pynutil.insert("extension:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + extension
            + pynutil.insert('"')
        )

        # Defines the final graph
        graph_telephone = (
            (graph_country_code + pynini.accep(NEMO_SPACE)).ques
            + graph_number_part
            + (pynini.accep(NEMO_SPACE) + graph_extension).ques
        )

        telephone_graph = graph_telephone.optimize()
        final_graph = self.add_tokens(telephone_graph)
        self.fst = final_graph
