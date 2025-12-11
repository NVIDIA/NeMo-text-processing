# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    delete_preserve_order,
    delete_space,
    extract_field,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese time.

    Converts tagged time entities into spoken form, e.g.:
    - time { hours: "tám" minutes: "ba mươi" } -> tám giờ ba mươi phút
    - time { hours: "mười bốn" minutes: "mười lăm" } -> mười bốn giờ mười lăm phút
    - time { hours: "chín" } -> chín giờ
    - time { minutes: "ba" seconds: "hai mươi" } -> ba phút hai mươi giây
    - time { hours: "tám" minutes: "hai mươi ba" zone: "g m t" } -> tám giờ hai mươi ba phút GMT

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        time_zone = convert_space(pynini.string_file(get_abs_path("data/time/time_zones.tsv")))

        # Extract components
        hour_component = extract_field("hours")
        timezone_component = extract_field("zone") @ time_zone

        # Handle zero and non-zero components
        zero_minute_component = pynutil.delete("minutes:") + delete_space + pynutil.delete("\"không\"")
        zero_second_component = pynutil.delete("seconds:") + delete_space + pynutil.delete("\"không\"")

        non_zero_minute_component = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE - pynini.accep("không"))
            + pynutil.delete("\"")
        )
        non_zero_second_component = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE - pynini.accep("không"))
            + pynutil.delete("\"")
        )

        # Components with units
        hour_with_unit = hour_component + pynutil.insert(" giờ")
        minute_with_unit = non_zero_minute_component + pynutil.insert(" phút")
        second_with_unit = non_zero_second_component + pynutil.insert(" giây")

        # Optional components
        optional_timezone = pynini.closure(delete_space + pynutil.insert(NEMO_SPACE) + timezone_component, 0, 1)
        optional_preserve_order = pynini.closure(delete_space + delete_preserve_order, 0, 1)

        # Pattern 1: hours + optional zero minutes/seconds + optional timezone
        pattern_hours_only = (
            hour_with_unit
            + pynini.closure(delete_space + zero_minute_component, 0, 1)
            + pynini.closure(delete_space + zero_second_component, 0, 1)
            + optional_timezone
            + optional_preserve_order
        )

        # Pattern 2: hours + minutes + optional zero seconds + optional timezone
        pattern_hours_minutes = (
            hour_with_unit
            + delete_space
            + pynutil.insert(NEMO_SPACE)
            + minute_with_unit
            + pynini.closure(delete_space + zero_second_component, 0, 1)
            + optional_timezone
            + optional_preserve_order
        )

        # Pattern 3: hours + zero minutes + seconds + optional timezone
        pattern_hours_seconds = (
            hour_with_unit
            + delete_space
            + zero_minute_component
            + delete_space
            + pynutil.insert(NEMO_SPACE)
            + second_with_unit
            + optional_timezone
            + optional_preserve_order
        )

        # Pattern 4: hours + minutes + seconds + optional timezone
        pattern_hours_minutes_seconds = (
            hour_with_unit
            + delete_space
            + pynutil.insert(NEMO_SPACE)
            + minute_with_unit
            + delete_space
            + pynutil.insert(NEMO_SPACE)
            + second_with_unit
            + optional_timezone
            + optional_preserve_order
        )

        # Pattern 5: minutes only + optional zero seconds
        pattern_minutes_only = minute_with_unit + pynini.closure(delete_space + zero_second_component, 0, 1)

        # Pattern 6: minutes + seconds
        pattern_minutes_seconds = minute_with_unit + delete_space + pynutil.insert(NEMO_SPACE) + second_with_unit

        # Pattern 7: seconds only
        pattern_seconds_only = second_with_unit

        patterns = [
            pattern_hours_only,
            pattern_hours_minutes,
            pattern_hours_seconds,
            pattern_hours_minutes_seconds,
            pattern_minutes_only,
            pattern_minutes_seconds,
            pattern_seconds_only,
        ]

        final_graph = pynini.union(*patterns)

        if not deterministic:
            # Add special case for half hour ("rưỡi")
            half_hour = (
                pynutil.delete("minutes:") + delete_space + pynutil.delete("\"ba mươi\"") + pynutil.insert("rưỡi")
            )
            half_hour_pattern = (
                hour_with_unit
                + delete_space
                + pynutil.insert(NEMO_SPACE)
                + half_hour
                + optional_timezone
                + optional_preserve_order
            )
            self.graph = pynini.union(final_graph, half_hour_pattern)
        else:
            self.graph = final_graph

        # Remove zero minutes and seconds from output
        remove_zero_minutes = pynini.cdrewrite(pynutil.delete(" không phút"), "", "", pynini.closure(NEMO_NOT_QUOTE))
        remove_zero_seconds = pynini.cdrewrite(pynutil.delete(" không giây"), "", "", pynini.closure(NEMO_NOT_QUOTE))

        self.fst = (
            self.delete_tokens(self.graph + optional_preserve_order).optimize()
            @ remove_zero_minutes
            @ remove_zero_seconds
        )
