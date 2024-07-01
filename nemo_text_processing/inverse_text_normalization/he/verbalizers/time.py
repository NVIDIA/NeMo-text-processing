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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        hour_to_noon = pynini.string_map([
            ("1", "13"),
            ("2", "14"),
            ("3", "15"),
            ("4", "16"),
            ("5", "17"),
            ("6", "18"),
        ])

        hour_to_evening = pynini.string_map([
            ("5", "17"),
            ("6", "18"),
            ("7", "19"),
            ("8", "20"),
            ("9", "21"),
            ("10", "22"),
            ("11", "23"),
        ])
        hour_to_night = pynini.string_map([
            ("8", "20"),
            ("9", "21"),
            ("10", "22"),
            ("11", "23"),
            ("12", "0"),
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
            ("4", "4"),
        ])

        day_suffixes = pynutil.delete("suffix: \"בוקר\"")
        noon_suffixes = pynutil.delete("suffix: \"צהריים\"")
        evening_suffixes = pynutil.delete("suffix: \"ערב\"")
        night_suffixes = pynutil.delete("suffix: \"לילה\"")

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        optional_suffix = pynini.closure(delete_space + day_suffixes, 0, 1)
        graph = hour + delete_space + pynutil.insert(":") + minute + optional_suffix

        for hour_to, suffix in zip([hour_to_noon, hour_to_evening, hour_to_night], [noon_suffixes, evening_suffixes, night_suffixes]):
            graph |= (
                hour @ hour_to
                + delete_space
                + pynutil.insert(":")
                + minute
                + delete_space
                + suffix
            )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

if __name__ == "__main__":
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    time = TimeFst().fst
    apply_fst('time { hours: "2" minutes: "10" }', time)
    apply_fst('time { hours: "2" minutes: "01" }', time)
    apply_fst('time { hours: "12" minutes: "03" }', time)
    apply_fst('time { hours: "2" minutes: "20" }', time)
    apply_fst('time { hours: "3" minutes: "00" suffix: "צהריים" }', time)
    apply_fst('time { hours: "2" minutes: "55" suffix: "לילה" }', time)
    apply_fst('time { hours: "2" minutes: "57" suffix: "בוקר" }', time)
    apply_fst('time { hours: "6" minutes: "32" suffix: "ערב" }', time)
