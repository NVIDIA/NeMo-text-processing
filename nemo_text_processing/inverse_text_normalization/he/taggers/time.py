# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst, delete_and
from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path, integer_to_text
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    delete_extra_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time in Hebrew.
    Conversion is made only when am / pm time is not ambiguous!
        e.g. שלוש דקות לחצות -> time { minutes: "57" hours: "23" }
        e.g. באחת ושתי דקות בצהריים -> time { morphosyntactic_features: "ב" hours: "1" minutes: "02" suffix: "צהריים" }
        e.g. שתיים ועשרה בבוקר -> time { hours: "2" minutes: "10" suffix: "בוקר" }
        e.g. שתיים ועשרה בצהריים -> time { hours: "2" minutes: "10" suffix: "צהריים" }
        e.g. שתיים עשרה ושלוש דקות אחרי הצהריים -> time { hours: "12" minutes: "03" suffix: "צהריים" }
        e.g. רבע לשש בערב -> time { minutes: "45" hours: "5" suffix: "ערב" }

    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        # hours, minutes, seconds, suffix, zone, style, speak_period
        midnight_to_hour_graph = pynini.string_file(get_abs_path("data/time/midnight_to_hour.tsv"))
        to_hour_graph = pynini.string_file(get_abs_path("data/time/to_hour.tsv"))

        minute_verbose_graph = pynini.string_file(get_abs_path("data/time/minute_verbose.tsv"))
        minute_to_graph = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))
        minute_to_verbose_graph = pynini.string_file(get_abs_path("data/time/minute_to_verbose.tsv"))

        suffix_graph = pynini.union(
            pynini.string_file(get_abs_path("data/time/day_suffix.tsv")),
            pynini.string_file(get_abs_path("data/time/noon_suffix.tsv")),
            pynini.string_file(get_abs_path("data/time/evening_suffix.tsv")),
            pynini.string_file(get_abs_path("data/time/night_suffix.tsv")),
        )

        time_prefix = pynini.string_file(get_abs_path("data/prefix.tsv"))
        time_prefix_graph = (
            pynutil.insert('morphosyntactic_features: "') + time_prefix + pynutil.insert('"') + insert_space
        )
        optional_time_prefix_graph = pynini.closure(time_prefix_graph, 0, 1)

        # only used for < 1000 thousand -> 0 weight
        cardinal = pynutil.add_weight(CardinalFst().graph_no_exception, weight=-0.7)

        labels_hour = [integer_to_text(x, only_fem=True)[0] for x in range(1, 13)]
        labels_minute_single = [integer_to_text(x, only_fem=True)[0] for x in range(2, 10)]
        labels_minute_double = [integer_to_text(x, only_fem=True)[0] for x in range(10, 60)]

        graph_hour = pynini.union(*labels_hour) @ cardinal
        graph_hour |= midnight_to_hour_graph
        add_leading_zero_to_double_digit = pynutil.insert("0") + NEMO_DIGIT
        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal @ add_leading_zero_to_double_digit
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal

        final_graph_hour = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')

        graph_minute = pynini.union(pynutil.insert("00"), graph_minute_single, graph_minute_double)

        final_suffix = pynutil.insert('suffix: "') + suffix_graph + pynutil.insert('"')
        final_suffix = delete_space + insert_space + final_suffix

        time_word = "דקות"
        optional_delete_time = pynini.closure(delete_space + pynutil.delete(time_word), 0, 1)
        graph_h_and_m = (
            final_graph_hour
            + delete_space
            + delete_and
            + insert_space
            + pynutil.insert('minutes: "')
            + pynini.union(graph_minute_single, graph_minute_double, minute_verbose_graph)
            + pynutil.insert('"')
            + optional_delete_time
        )

        graph_special_m_to_h_suffix_time = (
            pynutil.insert('minutes: "')
            + minute_to_verbose_graph
            + pynutil.insert('"')
            + delete_space
            + pynutil.delete("ל")
            + insert_space
            + pynutil.insert('hours: "')
            + to_hour_graph
            + pynutil.insert('"')
        )

        graph_m_to_h_suffix_time = (
            pynutil.insert('minutes: "')
            + pynini.union(graph_minute_single, graph_minute_double) @ minute_to_graph
            + pynutil.insert('"')
            + optional_delete_time
            + delete_space
            + pynutil.delete("ל")
            + insert_space
            + pynutil.insert('hours: "')
            + to_hour_graph
            + pynutil.insert('"')
        )

        graph_h = (
            optional_time_prefix_graph
            + delete_zero_or_one_space
            + final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + (pynutil.insert("00") | graph_minute)
            + pynutil.insert('"')
            + final_suffix
        )

        midnight_graph = (
            optional_time_prefix_graph
            + delete_zero_or_one_space
            + pynutil.insert('hours: "')
            + midnight_to_hour_graph
            + pynutil.insert('"')
            + insert_space
            + pynutil.insert('minutes: "')
            + (pynutil.insert("00") | graph_minute)
            + pynutil.insert('"')
        )

        graph_midnight_and_m = (
            pynutil.insert('hours: "')
            + midnight_to_hour_graph
            + pynutil.insert('"')
            + delete_space
            + delete_and
            + insert_space
            + pynutil.insert('minutes: "')
            + pynini.union(graph_minute_single, graph_minute_double, minute_verbose_graph)
            + pynutil.insert('"')
            + optional_delete_time
        )

        to_midnight_verbose_graph = (
            pynutil.insert('minutes: "')
            + minute_to_verbose_graph
            + pynutil.insert('"')
            + delete_space
            + pynutil.delete("ל")
            + insert_space
            + pynutil.insert('hours: "')
            + to_hour_graph
            + pynutil.insert('"')
        )

        graph_m_to_midnight = (
            pynutil.insert('minutes: "')
            + pynini.union(graph_minute_single, graph_minute_double) @ minute_to_graph
            + pynutil.insert('"')
            + optional_delete_time
            + delete_space
            + pynutil.delete("ל")
            + insert_space
            + pynutil.insert('hours: "')
            + to_hour_graph
            + pynutil.insert('"')
        )

        final_graph_midnight = (
            optional_time_prefix_graph
            + delete_zero_or_one_space
            + (midnight_graph | to_midnight_verbose_graph | graph_m_to_midnight | graph_midnight_and_m)
        )

        final_graph = (
            optional_time_prefix_graph
            + delete_zero_or_one_space
            + (graph_h_and_m | graph_special_m_to_h_suffix_time | graph_m_to_h_suffix_time)
            + final_suffix
        )
        final_graph |= graph_h
        final_graph |= final_graph_midnight

        final_graph = self.add_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
