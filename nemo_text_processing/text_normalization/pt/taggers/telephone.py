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

from nemo_text_processing.text_normalization.pt.graph_utils import NEMO_SPACE, NEMO_WHITE_SPACE, GraphFst, insert_space
from nemo_text_processing.text_normalization.pt.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying pt-BR telephone and IP formats, e.g.
        (11) 99999-8888 -> telephone { number_part: "um um nove nove nove nove nove oito oito oito oito" }
        +55 11 3333-4444 -> telephone { country_code: "mais cinco cinco" number_part: "um um três três três três quatro quatro quatro quatro" }
        192.168.1.1 -> telephone { number_part: "um nove dois ponto um seis oito ponto um ponto um" }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()
        single_digits = (digit | zero).optimize()

        # Only strip grouping punctuation between digit blocks; do not delete spaces
        # (deleting spaces would glue spoken digit words together in the output).
        separators = pynini.union("-", ".")
        delete_separator = pynini.closure(pynutil.delete(separators), 0, 1)
        delete_optional_spaces = pynini.closure(pynutil.delete(NEMO_WHITE_SPACE), 0)

        def n_digits(n: int):
            return pynini.closure(single_digits + insert_space, n - 1, n - 1) + single_digits

        country_digits = n_digits(1) | n_digits(2) | n_digits(3)
        country_code = pynini.cross("+", "mais ") + country_digits

        ip_prompts = pynini.string_file(get_abs_path("data/telephone/ip_prompt.tsv"))
        telephone_prompts = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        tel_prompt_sequence = telephone_prompts + NEMO_SPACE + pynini.closure(country_code, 0, 1)

        country_code_graph = (
            pynutil.insert('country_code: "')
            + (country_code | ip_prompts | tel_prompt_sequence)
            + delete_separator
            + pynutil.insert('"')
        )

        area_code = (pynutil.delete("(") + n_digits(2) + pynutil.delete(")")) | n_digits(2)

        eleven_digit_graph = (
            area_code
            + delete_optional_spaces
            + insert_space
            + n_digits(5)
            + delete_separator
            + insert_space
            + n_digits(4)
        )
        ten_digit_graph = (
            area_code
            + delete_optional_spaces
            + insert_space
            + n_digits(4)
            + delete_separator
            + insert_space
            + n_digits(4)
        )
        nine_digit_graph = n_digits(5) + delete_separator + insert_space + n_digits(4)
        eight_digit_graph = n_digits(4) + delete_separator + insert_space + n_digits(4)
        seven_digit_graph = n_digits(3) + delete_separator + insert_space + n_digits(4)

        digit_to_str_graph = single_digits + pynini.closure(pynutil.insert(" ") + single_digits, 0, 2)
        ip_graph = digit_to_str_graph + (pynini.cross(".", " ponto ") + digit_to_str_graph) ** 3

        number_part = (
            eleven_digit_graph
            | ten_digit_graph
            | nine_digit_graph
            | eight_digit_graph
            | seven_digit_graph
            | pynutil.add_weight(ip_graph, 0.01)
        )
        number_part = pynutil.insert('number_part: "') + number_part + pynutil.insert('"')

        extension_prompt = pynini.string_file(get_abs_path("data/telephone/extension_prompt.tsv"))
        delete_ext = pynini.cross(pynini.project(extension_prompt, "input"), "")
        ext_graph = (
            pynutil.insert('extension: "')
            + delete_optional_spaces
            + delete_ext
            + delete_optional_spaces
            + pynutil.insert("ramal ")
            + n_digits(1)
            + pynini.closure(insert_space + n_digits(1), 0, 3)
            + pynutil.insert('"')
        )

        graph = (
            pynini.closure(country_code_graph + delete_optional_spaces + insert_space, 0, 1)
            + number_part
            + pynini.closure(delete_optional_spaces + insert_space + ext_graph, 0, 1)
        )

        self.fst = self.add_tokens(graph).optimize()
