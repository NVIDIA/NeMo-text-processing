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
    NEMO_DIGIT,
    NEMO_SPACE,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone.
    The grammar assumes telephone number formatting used in Germany and Austria (DIN 5008).
    For more info go to: https://de.wikipedia.org/wiki/DIN_5008
        telephone { number_part: "112" } -> 112
        telephone { country_code: "015" number_part: "25 673421" } -> 015 25 673421
        telephone { country_code: "245" number_part: "141410" } -> 245 141410
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        # Handles country codes
        country_code_plus = pynini.accep("+") + pynini.closure(NEMO_DIGIT, 1, 3)
        country_code_three_digit = pynini.closure(NEMO_DIGIT, 1, 3)

        graph_country_code = (
            pynutil.delete("country_code:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + (country_code_plus | country_code_three_digit)
            + pynutil.delete('"')
        )

        # Handles the main part of the phone number
        optional_area_code = pynini.closure(NEMO_DIGIT, 1, 3)
        main_number_part = pynini.closure(NEMO_DIGIT, 1, 6)
        emergency_codes = pynini.closure(NEMO_DIGIT, 3, 3)
        main_number_section = (
            (optional_area_code + pynini.accep(NEMO_SPACE)).ques + main_number_part
        ) | emergency_codes

        graph_number_part = (
            pynutil.delete("number_part:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + main_number_section
            + pynutil.delete('"')
        )

        # Handles extensions
        hyphen = pynini.accep("-")
        extension = hyphen + pynini.closure(NEMO_DIGIT, 1, 2)

        graph_extension = (
            pynutil.delete("extension:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + extension
            + pynutil.delete('"')
        )

        graph_telephone = (
            (graph_country_code + pynini.accep(NEMO_SPACE)).ques
            + graph_number_part
            + (pynini.accep(NEMO_SPACE) + graph_extension).ques
        )

        delete_tokens = self.delete_tokens(graph_telephone)
        self.fst = delete_tokens.optimize()
