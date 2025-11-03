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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. hai rưỡi -> time { hours: "2" minutes: "30" }
        e.g. chín giờ kém hai mươi -> time { hours: "8" minutes: "40" }
        e.g. ba phút hai giây -> time { minutes: "3" seconds: "2" }
        e.g. mười giờ chín phút bốn mươi lăm giây -> time { hours: "10" minutes: "9" seconds: "45" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        graph_hours_to = pynini.string_file(get_abs_path("data/time/hours_to.tsv"))
        graph_minutes_to = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))
        graph_hours = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        graph_minutes = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
        time_zone_graph = pynini.invert(pynini.string_file(get_abs_path("data/time/time_zone.tsv")))

        graph_half = pynini.cross("rưỡi", "30")
        oclock = pynini.cross("giờ", "")
        minute = pynini.cross("phút", "")
        optional_minute = pynini.closure(delete_space + minute, 0, 1)
        second = pynini.cross("giây", "")

        # Zero prefix patterns for minutes (linh, lẻ, không)
        # Examples: "linh năm" -> "05", "không tám" -> "08"
        zero_prefix = pynini.string_file(get_abs_path("data/time/zero_prefix.tsv"))
        graph_zero_minute = zero_prefix + delete_space + graph_minutes
        graph_minute_extended = graph_minutes | graph_zero_minute

        final_graph_hour = pynutil.insert('hours: "') + graph_hours + pynutil.insert('"') + delete_space + oclock
        graph_minute = graph_minute_extended + optional_minute
        graph_second = graph_minute_extended + delete_space + second

        # Optional time zone support
        final_time_zone_optional = pynini.closure(
            delete_space
            + insert_space
            + pynutil.insert('zone: "')
            + convert_space(time_zone_graph)
            + pynutil.insert('"'),
            0,
            1,
        )

        # Time pattern combinations
        # Pattern 1: Hour + Minutes (e.g., "tám giờ hai mươi" -> 8:20)
        graph_hm = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + (graph_minute | graph_half)
            + pynutil.insert('"')
        )

        # Pattern 2: Hour + Minutes + Seconds (e.g., "tám giờ hai mươi phút ba mươi giây" -> 8:20:30)
        graph_hms = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minutes
            + delete_space
            + minute
            + pynutil.insert('"')
            + delete_extra_space
            + pynutil.insert('seconds: "')
            + graph_second
            + pynutil.insert('"')
        )

        # Pattern 3: Minutes + Seconds only (e.g., "ba phút hai mươi giây" -> 3p20s)
        graph_ms = (
            pynutil.insert('minutes: "')
            + graph_minutes
            + delete_space
            + minute
            + pynutil.insert('"')
            + delete_extra_space
            + pynutil.insert('seconds: "')
            + (graph_second | graph_half)
            + pynutil.insert('"')
        )

        # Pattern 4: Hour + Seconds only (e.g., "ba giờ mười giây" -> 3:00:10)
        graph_hs = (
            final_graph_hour
            + delete_extra_space
            + pynutil.insert('minutes: "0"')
            + delete_extra_space
            + pynutil.insert('seconds: "')
            + graph_second
            + pynutil.insert('"')
        )

        # "Kém" pattern components (e.g., "chín giờ kém hai mươi" -> 8:40)
        graph_hours_to_component = graph_hours @ graph_hours_to
        graph_minutes_to_component = graph_minutes @ graph_minutes_to

        # Pattern 5: "Kém" time format (hour minus minutes)
        graph_time_to = (
            pynutil.insert('hours: "')
            + graph_hours_to_component
            + pynutil.insert('"')
            + delete_space
            + oclock
            + delete_space
            + pynutil.delete("kém")
            + delete_extra_space
            + pynutil.insert('minutes: "')
            + graph_minutes_to_component
            + pynutil.insert('"')
            + optional_minute
        )

        # Combine all time patterns
        final_graph = (
            pynini.union(
                final_graph_hour,  # Hour only
                graph_hm,  # Hour + Minutes
                graph_hms,  # Hour + Minutes + Seconds
                graph_hs,  # Hour + Seconds
                graph_ms,  # Minutes + Seconds only
                graph_time_to,  # "Kém" pattern
            )
            + final_time_zone_optional
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
