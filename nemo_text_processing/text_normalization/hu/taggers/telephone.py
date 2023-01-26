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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hu.graph_utils import ensure_space
from nemo_text_processing.text_normalization.hu.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hu.utils import get_abs_path
from pynini.lib import pynutil


class TelephoneFst(GraphFst):
    """
    tel: + (36) 1 441-4000
    Finite state transducer for classifying telephone numbers, e.g.
        + (36) 1 441-4000 -> { number_part: "plusz harminchat egy négyszáznegyvenegy négyezer" }.

    Hungarian numbers are written in the following formats:
        06 1 XXX XXXX
        06 AA XXX-XXX
        06 AA XXX-XXXX (mobile phones)

    See:
        https://en.wikipedia.org/wiki/Telephone_numbers_in_Hungary

    Args:
		deterministic: if True will provide a single transduction option,
			for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)
        cardinal = CardinalFst(deterministic)
        area_codes = pynini.string_file(get_abs_path("data/telephone/area_codes.tsv")) @ cardinal.graph
        country_codes = pynini.string_file(get_abs_path("data/telephone/country_codes.tsv")) @ cardinal.graph
        self.country_codes = country_codes.optimize()

        add_separator = pynutil.insert(", ")

        zero_space = cardinal.zero_space
        digit = cardinal.digit
        two_digits = cardinal.two_digits_read
        three_digits = cardinal.three_digits_read
        four_digits = cardinal.four_digits_read

        separators = pynini.union(
            NEMO_SPACE,
            pynini.cross("-", " ")
        )
        area_separators = pynini.union(
            separators,
            pynini.cross("/", " ")
        )

        zero = pynini.cross("0", "nulla")
        digit |= zero

        special_numbers = pynini.string_file(get_abs_path("data/telephone/special_numbers.tsv"))

        telephone_abbr = pynini.string_file(get_abs_path("data/telephone/telephone_abbr.tsv"))
        telephone_prompt = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        prompt = pynutil.insert("prompt: \"") + telephone_prompt + pynutil.insert("\"")
        prompt |= pynutil.insert("prompt: \"") + telephone_abbr + pynutil.insert("\"")
        prompt |= pynutil.insert("prompt: \"") + telephone_prompt + NEMO_SPACE + telephone_abbr + pynutil.insert("\"")

        plus = pynini.cross("+", "plusz ")
        plus |= pynini.cross("00", "nulla nulla ")

        country = pynini.closure(pynutil.delete("("), 0, 1) + country_codes + pynini.closure(pynutil.delete(")"), 0, 1)
        country = plus + pynini.closure(pynutil.delete(" "), 0, 1) + country
        country_code = pynutil.insert("country_code: \"") + country + pynutil.insert("\"")

        trunk = "06" @ cardinal.two_digits_read


        area_part = area_codes + area_separators
        area_part |= bracketed + add_separator

        base_number_part = pynini.union(
            three_digits + separators + three_digits,
            three_digits + separators + two_digits + separators + two_digits,
            three_digits + separators + four_digits,
            four_digits + separators + three_digits,
            two_digits + separators + two_digits + separators + three_digits,
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
        prompt = prompt + prompt_pass
        graph = pynini.union(
            country_code + ensure_space + number_part,
            country_code + ensure_space + number_part + ext_prompt + extension,
            number_part + ext_prompt + extension,
            prompt + number_part,
            prompt + special_numbers,
            prompt + country_code + number_part,
            prompt + country_code + number_part + ext_prompt + extension,
            prompt + number_part + ext_prompt + extension,
        )
        self.tel_graph = graph.optimize()

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
