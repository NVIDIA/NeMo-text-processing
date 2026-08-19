# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.sv.graph_utils import ensure_space
from nemo_text_processing.text_normalization.sv.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.sv.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    tfn. 08-789 52 25
    Finite state transducer for classifying telephone numbers, e.g.
        123-123-5678 -> { number_part: "ett två tre ett två tre fyra sex sju åtta" }.

    Swedish numbers are written in the following formats:
        0X-XXX XXX XX
        0X-XXX XX XX
        0X-XX XX XX
        0XX-XXX XX XX
        0XX-XX XX XX
        0XX-XXX XX
        0XXX-XX XX XX
        0XXX-XXX XX

    See:
        https://en.wikipedia.org/wiki/National_conventions_for_writing_telephone_numbers#Sweden
        https://codegolf.stackexchange.com/questions/195787/format-a-swedish-phone-number

    Args:
                deterministic: if True will provide a single transduction option,
                        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)
        cardinal = CardinalFst(deterministic)
        add_separator = pynutil.insert(", ")
        zero_space = cardinal.zero_space
        digit = cardinal.digit
        two_digits = cardinal.two_digits_read
        three_digits = cardinal.three_digits_read
        two_or_three_digits = (two_digits | three_digits).optimize()
        one_two_or_three_digits = (digit | two_or_three_digits).optimize()
        zero_after_country_code = pynini.union(pynini.cross("(0)", "noll "), zero_space)
        bracketed = pynutil.delete("(") + one_two_or_three_digits + pynutil.delete(")")

        zero = pynini.cross("0", "noll")
        digit |= zero

        special_numbers = pynini.string_file(get_abs_path("data/telephone/special_numbers.tsv"))

        passable = pynini.union(":", ": ", " ")
        prompt_pass = pynini.closure(pynutil.delete(passable) + insert_space, 0, 1)
        telephone_abbr = pynini.string_file(get_abs_path("data/telephone/telephone_abbr.tsv"))
        telephone_abbr = telephone_abbr + prompt_pass
        telephone_prompt = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        prompt_as_code = pynutil.insert("country_code: \"") + telephone_prompt + pynutil.insert("\"")
        prompt_as_code |= pynutil.insert("country_code: \"") + telephone_abbr + pynutil.insert("\"")
        prompt_as_code |= (
            pynutil.insert("country_code: \"") + telephone_prompt + NEMO_SPACE + telephone_abbr + pynutil.insert("\"")
        )
        prompt_inner = telephone_prompt | telephone_abbr
        prompt_inner |= telephone_prompt + NEMO_SPACE + telephone_abbr

        country = pynini.closure(pynini.cross("+", "plus "), 0, 1) + one_two_or_three_digits
        country_code = pynutil.insert("country_code: \"") + country + pynutil.insert("\"")
        country_code |= prompt_as_code
        country_code |= pynutil.insert("country_code: \"") + prompt_inner + NEMO_SPACE + country + pynutil.insert("\"")

        opt_dash = pynini.closure(pynutil.delete("-"), 0, 1)
        area_part = zero_after_country_code + one_two_or_three_digits + opt_dash + add_separator
        area_part |= bracketed + add_separator

        base_number_part = pynini.union(
            three_digits + NEMO_SPACE + three_digits + NEMO_SPACE + two_digits,
            three_digits + NEMO_SPACE + two_digits + NEMO_SPACE + two_digits,
            three_digits + NEMO_SPACE + two_digits + insert_space + two_digits,
            two_digits + NEMO_SPACE + two_digits + NEMO_SPACE + two_digits,
            two_digits + NEMO_SPACE + two_digits + insert_space + two_digits,
            three_digits + NEMO_SPACE + two_digits,
        )
        number_part = area_part + delete_space + base_number_part

        self.number_graph = number_part
        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")
        extension = pynutil.insert("extension: \"") + one_two_or_three_digits + pynutil.insert("\"")
        extension = pynini.closure(insert_space + extension, 0, 1)
        ext_prompt = NEMO_SPACE + pynutil.delete(pynini.union("ankn", "ankn.", "anknytning")) + ensure_space
        passable = pynini.union(":", ": ", " ")
        prompt_pass = pynutil.delete(passable) + insert_space

        special_numbers = pynutil.insert("number_part: \"") + special_numbers + pynutil.insert("\"")
        graph = pynini.union(
            country_code + ensure_space + number_part,
            country_code + ensure_space + number_part + ext_prompt + extension,
            number_part + ext_prompt + extension,
            country_code + number_part,
            country_code + special_numbers,
            country_code + number_part + ext_prompt + extension,
        )
        self.tel_graph = graph.optimize()

        # No need to be so exact here, but better for ITN to have it
        three_digit_area_code_digit_two = pynini.union("1", "2", "3", "4", "7")
        three_digit_area_code_no_zero = (three_digit_area_code_digit_two + NEMO_DIGIT) @ cardinal.two_digits_read
        three_digit_area_code = zero_space + three_digit_area_code_no_zero
        four_digit_area_code_digit_two = pynini.union("5", "6", "9")
        four_digit_area_code_no_zero = (four_digit_area_code_digit_two + NEMO_DIGIT) @ cardinal.three_digits_read
        four_digit_area_code = zero_space + four_digit_area_code_no_zero
        two_digit_area_code = "08" @ cardinal.two_digits_read
        self.area_codes = two_digit_area_code | three_digit_area_code | four_digit_area_code
        self.area_codes_no_zero = (
            three_digit_area_code_no_zero | four_digit_area_code_no_zero | pynini.cross("8", "åtta")
        )
        country_code_lead = pynini.cross("+", "plus") | pynini.cross("00", "noll noll")
        raw_country_codes = pynini.string_file(get_abs_path("data/telephone/country_codes.tsv"))
        self.country_code = country_code_lead + insert_space + (raw_country_codes @ cardinal.any_read_digit)
        self.country_plus_area_code = self.country_code + NEMO_SPACE + self.area_codes_no_zero

        # ip
        ip_prompts = pynini.string_file(get_abs_path("data/telephone/ip_prompt.tsv"))
        ip_graph = one_two_or_three_digits + (pynini.cross(".", " punkt ") + one_two_or_three_digits) ** 3
        graph |= (
            pynini.closure(
                pynutil.insert("country_code: \"") + ip_prompts + pynutil.insert("\"") + delete_extra_space, 0, 1
            )
            + pynutil.insert("number_part: \"")
            + ip_graph.optimize()
            + pynutil.insert("\"")
        )
        # ssn
        ssn_prompts = pynini.string_file(get_abs_path("data/telephone/ssn_prompt.tsv"))
        four_digit_part = digit + (pynutil.insert(" ") + digit) ** 3
        ssn_separator = pynini.cross("-", ", ")
        ssn_graph = three_digits + ssn_separator + two_digits + ssn_separator + four_digit_part

        graph |= (
            pynini.closure(
                pynutil.insert("country_code: \"") + ssn_prompts + pynutil.insert("\"") + delete_extra_space, 0, 1
            )
            + pynutil.insert("number_part: \"")
            + ssn_graph.optimize()
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
