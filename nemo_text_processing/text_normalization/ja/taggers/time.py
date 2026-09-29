# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        1時30分 -> time { hours: "一" minutes: "三十" }
        今夜0時 -> time { suffix: "今夜" hours: "零" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.just_cardinals

        hour_clock = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        minute_second_clock = pynini.string_file(get_abs_path("data/time/minute_second.tsv"))
        division = pynini.string_file(get_abs_path("data/time/division.tsv"))
        zero_decimal = pynini.string_file(get_abs_path("data/numbers/zero_decimal.tsv"))
        decimal_point = pynini.string_file(get_abs_path("data/numbers/decimal_point.tsv"))
        suffixes = dict(load_labels(get_abs_path("data/time/suffix.tsv")))
        hour_suffix = suffixes["hour"]
        hour_variants = pynini.union(hour_suffix, suffixes["duration_hour"], suffixes["approximate_hour"])
        minute_suffix = suffixes["minute"]
        minute_modifier = pynini.union(suffixes["past"], suffixes["approximate"])
        half = suffixes["half"]
        second_suffix = suffixes["second"]

        division_component = pynutil.insert('suffix: "') + division + pynutil.insert('"')
        hour_number = pynutil.add_weight(zero_decimal, -0.1) | graph_cardinal
        hour_component = (
            pynutil.insert('hours: "')
            + (hour_number | (graph_cardinal + decimal_point + graph_cardinal))
            + hour_variants
            + pynutil.insert('"')
        )
        minute_component = pynutil.insert('minutes: "') + (
            graph_cardinal | (graph_cardinal + decimal_point + graph_cardinal)
        ) + pynini.accep(minute_suffix) + pynini.closure(minute_modifier, 0, 1) + pynutil.insert('"') | (
            pynutil.insert('minutes: "')
            + pynini.accep(half)
            + pynini.closure(minute_modifier, 0, 1)
            + pynutil.insert('"')
        )
        second_component = (
            pynutil.insert('seconds: "')
            + (graph_cardinal | (graph_cardinal + decimal_point + graph_cardinal))
            + pynini.accep(second_suffix)
            + pynutil.insert('"')
        )

        graph_individual_time = pynini.closure(division_component + pynutil.insert(" "), 0, 1) + (
            hour_component
            | minute_component
            | second_component
            | (hour_component + pynutil.insert(" ") + minute_component)
            | (hour_component + pynutil.insert(" ") + minute_component + pynutil.insert(" ") + second_component)
            | (minute_component + pynutil.insert(" ") + second_component)
        )

        colon = pynutil.delete(":")
        hour_clock_component = (
            pynutil.insert('hours: "')
            + pynutil.delete("0").ques
            + hour_clock
            + pynutil.insert(hour_suffix)
            + pynutil.insert('"')
        )
        minute_clock_component = (
            pynutil.insert('minutes: "')
            + pynutil.delete("0").ques
            + minute_second_clock
            + pynutil.insert(minute_suffix)
            + pynutil.insert('"')
        )
        second_clock_component = (
            pynutil.insert('seconds: "')
            + pynutil.delete("0").ques
            + minute_second_clock
            + pynutil.insert(second_suffix)
            + pynutil.insert('"')
        )

        graph_clock_with_seconds = (
            hour_clock_component
            + pynutil.insert(" ")
            + colon
            + minute_clock_component
            + pynutil.insert(" ")
            + colon
            + second_clock_component
        )
        graph_clock_with_minutes = hour_clock_component + pynutil.insert(" ") + colon + minute_clock_component
        graph_clock_without_minutes = hour_clock_component + colon + pynutil.delete("00")
        graph_clock = graph_clock_with_seconds | graph_clock_with_minutes | graph_clock_without_minutes

        graph = graph_individual_time | graph_clock

        graph_final = self.add_tokens(graph)
        self.fst = graph_final.optimize()
