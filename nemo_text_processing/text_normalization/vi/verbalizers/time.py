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
    GraphFst,
    convert_space,
    delete_preserve_order,
    delete_space,
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
        quoted_text = pynini.closure(NEMO_NOT_QUOTE)

        def extract_field(field_name):
            return (
                pynutil.delete(f"{field_name}:")
                + delete_space
                + pynutil.delete("\"")
                + quoted_text
                + pynutil.delete("\"")
            )

        hour_component = extract_field("hours")
        timezone_component = extract_field("zone") @ time_zone

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

        hour_with_unit = hour_component + pynutil.insert(" giờ")
        minute_with_unit = non_zero_minute_component + pynutil.insert(" phút")
        second_with_unit = non_zero_second_component + pynutil.insert(" giây")

        optional_timezone = pynini.closure(delete_space + pynutil.insert(" ") + timezone_component, 0, 1)
        optional_preserve_order = pynini.closure(delete_space + delete_preserve_order, 0, 1)

        patterns = [
            hour_with_unit
            + pynini.closure(delete_space + zero_minute_component, 0, 1)
            + pynini.closure(delete_space + zero_second_component, 0, 1)
            + optional_timezone
            + optional_preserve_order,
            hour_with_unit
            + delete_space
            + pynutil.insert(" ")
            + minute_with_unit
            + pynini.closure(delete_space + zero_second_component, 0, 1)
            + optional_timezone
            + optional_preserve_order,
            hour_with_unit
            + delete_space
            + zero_minute_component
            + delete_space
            + pynutil.insert(" ")
            + second_with_unit
            + optional_timezone
            + optional_preserve_order,
            hour_with_unit
            + delete_space
            + pynutil.insert(" ")
            + minute_with_unit
            + delete_space
            + pynutil.insert(" ")
            + second_with_unit
            + optional_timezone
            + optional_preserve_order,
            minute_with_unit + pynini.closure(delete_space + zero_second_component, 0, 1),
            minute_with_unit + delete_space + pynutil.insert(" ") + second_with_unit,
            second_with_unit,
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
                + pynutil.insert(" ")
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
