# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import ES_PLUS


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        uno dos tres uno dos tres cinco seis siete ocho -> { number_part: "123-123-5678" }.
        If 10 digits are spoken, they are grouped as 3+3+4 (eg. 123-456-7890).
        If 9 digits are spoken, they are grouped as 3+3+3 (eg. 123-456-789).
        If 8 digits are spoken, they are grouped as 4+4 (eg. 1234-5678).
        In Spanish, digits are generally spoken individually, or as 2-digit numbers,
        eg. "one twenty three" = "123",
            "twelve thirty four" = "1234".

        (we ignore more complicated cases such as "three hundred and two" or "three nines").

        Args:
            input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))
        graph_zero = pynini.cross("cero", "0")

        if input_case == INPUT_CASED:
            graph_digit = capitalized_input_graph(graph_digit)
            graph_ties = capitalized_input_graph(graph_ties)
            graph_teen = capitalized_input_graph(graph_teen)
            graph_twenties = capitalized_input_graph(graph_twenties)
            graph_zero = pynini.cross(pynini.union("cero", "Cero"), "0").optimize()

        single_digits = graph_digit.optimize() | graph_zero

        double_digits = pynini.union(
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + delete_space + pynutil.delete("y") + delete_space + graph_digit),
        )

        # self.single_digits = single_digits
        # self.double_digits = double_digits
        digit_twice = single_digits + pynutil.delete(" ") + single_digits
        digit_thrice = digit_twice + pynutil.delete(" ") + single_digits

        # accept `doble cero` -> `00` and `triple ocho` -> `888`
        digit_words = pynini.union(graph_digit.optimize(), graph_zero).invert()

        doubled_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_words, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_words, "output"),
                    pynutil.insert("doble ") + pynini.project(str(i) @ digit_words, "output"),
                )
                for i in range(10)
            ]
        )
        doubled_digit.invert()
        doubled_digit @= digit_twice

        tripled_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_words, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_words, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_words, "output"),
                    pynutil.insert("triple ") + pynini.project(str(i) @ digit_words, "output"),
                )
                for i in range(10)
            ]
        )
        tripled_digit.invert()
        tripled_digit @= digit_thrice

        # Denormalized phone numbers are grouped in sets of 3 or 4 digits
        group_of_two = pynini.union(doubled_digit, digit_twice, double_digits)

        group_of_three = pynini.union(tripled_digit, single_digits + pynutil.delete(" ") + group_of_two,)

        group_of_four = pynini.union(
            group_of_two + pynutil.delete(" ") + group_of_two,
            tripled_digit + pynutil.delete(" ") + single_digits,
            single_digits + pynutil.delete(" ") + tripled_digit,
        )

        insert_separator = pynini.cross(" ", "-")

        # 10-digit option
        ten_digit_graph = group_of_three + insert_separator + group_of_three + insert_separator + group_of_four

        # 9-digit option
        nine_digit_graph = group_of_three + insert_separator + group_of_three + insert_separator + group_of_three

        # 8-digit option
        eight_digit_graph = group_of_four + insert_separator + group_of_four

        plus = pynini.accep("más")
        if input_case == INPUT_CASED:
            plus |= ES_PLUS

        # optionally denormalize country codes
        optional_country_code = pynini.closure(
            pynini.cross(plus, "+")
            + delete_space
            + (single_digits | group_of_two | group_of_three)
            + insert_separator,
            0,
            1,
        )

        ext_phrase = pynini.accep(" extensión ")
        if input_case == INPUT_CASED:
            ext_phrase = pynini.union(" extensión ", " Extensión ")
        # optionally denormalize extensions
        optional_extension = pynini.closure(
            pynini.cross(ext_phrase, " ext. ") + (single_digits | group_of_two | group_of_three), 0, 1
        )

        number_part = (
            optional_country_code
            + pynini.union(pynutil.add_weight(ten_digit_graph, -0.01), nine_digit_graph, eight_digit_graph)
            + optional_extension
        )

        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")

        graph = number_part
        if input_case == INPUT_CASED:
            graph |= capitalized_input_graph(graph)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
