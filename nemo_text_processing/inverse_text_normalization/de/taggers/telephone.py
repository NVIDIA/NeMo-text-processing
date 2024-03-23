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

from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_space


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g. 
        plus vier vier eins eins eins zwei drei vier eins zwei drei vier -> { number_part: "+44-111-234-1234" }.
        If 10 digits are spoken, they are grouped as 3+3+4 (eg. 123-456-7890).
        If 9 digits are spoken, they are grouped as 3+3+3 (eg. 123-456-789).
        If 8 digits are spoken, they are grouped as 4+4 (eg. 1234-5678).
        In german, digits are generally spoken individually, or rarely as 2-digit numbers,
        eg. "one twenty three" = "123",
            "twelve thirty four" = "1234".

        (we ignore more complicated cases such as "three hundred and two" or "three nines").
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        single_digits = graph_digit.optimize() | pynini.cross("null", "0")

        double_digits = pynini.union(
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + delete_space + pynutil.delete("y") + delete_space + graph_digit),
        )

        # self.single_digits = single_digits
        # self.double_digits = double_digits
        digit_twice = single_digits + pynutil.delete(" ") + single_digits
        digit_thrice = digit_twice + pynutil.delete(" ") + single_digits

        # accept `doble cero` -> `00` and `triple ocho` -> `888`
        digit_words = pynini.union(graph_digit.optimize(), pynini.cross("null", "0")).invert()

        doubled_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_words, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_words, "output"),
                    pynutil.insert("doppelt ") + pynini.project(str(i) @ digit_words, "output"),
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

        # optionally denormalize country codes
        optional_country_code = pynini.closure(
            pynini.cross("plus ", "+") + (single_digits | group_of_two | group_of_three) + insert_separator, 0, 1
        )

        # optionally denormalize extensions
        optional_extension = pynini.closure(
            pynini.cross(" extension ", " ext. ") + (single_digits | group_of_two | group_of_three), 0, 1
        )

        number_part = (
            optional_country_code
            + pynini.union(pynutil.add_weight(ten_digit_graph, -0.01), nine_digit_graph, eight_digit_graph)
            + optional_extension
        )

        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
