# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
    plurals,
)
from nemo_text_processing.text_normalization.sv.graph_utils import SV_ALPHA
from nemo_text_processing.text_normalization.sv.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.sv.utils import get_abs_path
from pynini.lib import pynutil


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
        zero_after_country_code = pynini.union(pynini.cross("(0)", "null "), zero_space)
        two_or_three_digits = (two_digits | three_digits).optimize()
        one_two_or_three_digits = (digit | two_or_three_digits).optimize()

        zero = pynini.cross("0", "null")
        digit |= zero

        special_numbers = pynini.string_file(get_abs_path("data/telephone/special_numbers.tsv"))

        telephone_abbr = pynini.string_file(get_abs_path("data/telephone/telephone_abbr.tsv"))
        telephone_prompt = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        prompt = pynutil.insert("prompt: \"") + telephone_prompt + pynutil.insert("\"")
        prompt |= pynutil.insert("prompt: \"") + telephone_abbr + pynutil.insert("\"")
        prompt |= pynutil.insert("prompt: \"") + telephone_prompt + NEMO_SPACE + telephone_abbr + pynutil.insert("\"")

        country_code = pynini.closure(pynini.cross("+", "plus "), 0, 1) + one_two_or_three_digits
        country_code = pynutil.insert("country_code: \"") + country_code + pynutil.insert("\"")
        country_code = country_code + pynini.closure(pynutil.delete("-"), 0, 1) + NEMO_SPACE

        area_part = pynini.cross("900", "niohundra")
        area_part |= one_two_or_three_digits

        area_part = (
            zero_space
            + (
                pynini.closure(pynutil.delete("("), 0, 1)
                + area_part
                + pynini.closure(
                    ((pynutil.delete(")") + pynini.closure(pynutil.delete(" "), 0, 1)) | pynutil.delete(")-")), 0, 1
                )
            )
        ) + add_separator

        number_words = pynini.union(
            three_digits + NEMO_SPACE + three_digits + NEMO_SPACE + two_digits,
            three_digits + NEMO_SPACE + two_digits + NEMO_SPACE + two_digits,
            two_digits + NEMO_SPACE + two_digits + NEMO_SPACE + two_digits,
            three_digits + NEMO_SPACE + two_digits,
            special_numbers,
        )

        number_part = area_part + number_words
        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")
        extension = (
            pynutil.insert("extension: \"") + pynini.closure(one_two_or_three_digits, 0, 3) + pynutil.insert("\"")
        )
        extension = pynini.closure(insert_space + extension, 0, 1)

        graph = plurals._priority_union(country_code + number_part, number_part, NEMO_SIGMA).optimize()
        graph = plurals._priority_union(country_code + number_part + extension, graph, NEMO_SIGMA).optimize()
        graph = plurals._priority_union(number_part + extension, graph, NEMO_SIGMA).optimize()
        graph = plurals._priority_union(prompt + country_code + number_part, number_part, NEMO_SIGMA).optimize()
        graph = plurals._priority_union(prompt + country_code + number_part + extension, graph, NEMO_SIGMA).optimize()
        graph = plurals._priority_union(prompt + number_part + extension, graph, NEMO_SIGMA).optimize()
        prompt

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
