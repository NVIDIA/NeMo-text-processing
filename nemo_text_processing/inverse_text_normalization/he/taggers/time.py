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
from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path, integer_to_text
from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
    NEMO_DIGIT,
    delete_and,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. twelve thirty -> time { hours: "12" minutes: "30" }
        e.g. twelve past one -> time { minutes: "12" hours: "1" }
        e.g. two o clock a m -> time { hours: "2" suffix: "a.m." }
        e.g. quarter to two -> time { hours: "1" minutes: "45" }
        e.g. quarter past two -> time { hours: "2" minutes: "15" }
        e.g. half past two -> time { hours: "2" minutes: "30" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period
        to_hour_graph = pynini.string_file(get_abs_path("data/time/to_hour.tsv"))
        minute_to_graph = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))
        suffix_graph = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))
        to_suffix_graph = pynini.union(
            pynutil.delete("לפנות") + delete_space,
            pynutil.delete("ב"),
            pynutil.delete("אחר") + delete_space + pynutil.delete("ה")
        )
        graph_minute_verbose = pynini.string_map([
            ("שלושת רבעי", "45"),
            ("חצי", "30"),
            ("רבע", "15"),
            ("עשרה", "10"),
            ("חמישה", "05"),
            ("דקה", "01"),
            ])
        graph_minute_to_verbose = pynini.string_map([
            ("רבע", "45"),
            ("עשרה", "50"),
            ("חמישה", "55"),
            ("דקה", "59"),
        ])

        # only used for < 1000 thousand -> 0 weight
        cardinal = pynutil.add_weight(CardinalFst().graph_no_exception, weight=-0.7)

        labels_hour = [integer_to_text(x, only_fem=True)[0] for x in range(1, 13)]
        labels_minute_single = [integer_to_text(x, only_fem=True)[0] for x in range(2, 10)]
        labels_minute_double = [integer_to_text(x, only_fem=True)[0] for x in range(10, 60)]

        graph_hour = pynini.union(*labels_hour) @ cardinal
        add_leading_zero_to_double_digit = pynutil.insert("0") + NEMO_DIGIT
        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal @ add_leading_zero_to_double_digit
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal

        final_graph_hour = pynutil.insert("hours: \"") + graph_hour + pynutil.insert("\"")
        graph_minute = pynini.union(
            pynutil.insert("00"),
            graph_minute_single,
            graph_minute_double
        )
        final_suffix = pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\"")
        final_suffix = delete_space + insert_space + to_suffix_graph + final_suffix
        final_suffix_optional = pynini.closure(final_suffix, 0, 1)

        graph_h_and_m = (
            final_graph_hour
            + delete_space
            + delete_and
            + insert_space
            + pynutil.insert("minutes: \"")
            + pynini.union(graph_minute_single, graph_minute_double, graph_minute_verbose)
            + pynutil.insert("\"")
            + (pynini.closure(delete_space + pynutil.delete("דקות"), 0, 1))
        )

        graph_special_m_to_h_suffix_time = (
            pynutil.insert("minutes: \"")
            + graph_minute_to_verbose
            + pynutil.insert("\"")
            + delete_space
            + pynutil.delete("ל")
            + insert_space
            + pynutil.insert("hours: \"")
            + to_hour_graph
            + pynutil.insert("\"")
        )

        graph_m_to_h_suffix_time = (
            pynutil.insert("minutes: \"")
            + pynini.union(graph_minute_single, graph_minute_double) @ minute_to_graph
            + pynutil.insert("\"")
            + pynini.closure(delete_space + pynutil.delete("דקות"), 0, 1)
            + delete_space
            + pynutil.delete("ל")
            + insert_space
            + pynutil.insert("hours: \"")
            + to_hour_graph
            + pynutil.insert("\"")
        )

        graph_h = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert("minutes: \"")
            + (pynutil.insert("00") | graph_minute)
            + pynutil.insert("\"")
            + final_suffix
        )
        final_graph = (
            (graph_h_and_m | graph_special_m_to_h_suffix_time | graph_m_to_h_suffix_time) + final_suffix_optional
        )
        final_graph |= graph_h

        final_graph = self.add_tokens(final_graph.optimize())

        self.fst = final_graph.optimize()


if __name__ == '__main__':
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst

    graph = TimeFst().fst
    apply_fst("שתיים ועשרה", graph)
    apply_fst("שתיים ודקה", graph)
    apply_fst("שתיים עשרה ושלוש דקות", graph)
    apply_fst("שתיים ועשרים דקות", graph)
    apply_fst("חמישה לשלוש", graph)
    apply_fst("שלוש בצהריים", graph)
    apply_fst("חמישה לשלוש בלילה", graph)
    apply_fst("חמישה לשלוש", graph)
    apply_fst("שלוש דקות לשלוש בבוקר", graph)
    # apply_fst("ארבע עשרה", graph) # should fail