# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time in Hebrew
        e.g. time { hours: "2" minutes: "55" suffix: "בלילה" } -> 2:55 בלילה
        e.g. time { hours: "2" minutes: "57" suffix: "בבוקר" } -> 2:57 בבוקר
        e.g. time { prefix: "ב" hours: "6" minutes: "32" suffix: "בערב" } -> ב-18:32 בערב
        e.g. time { prefix: "בשעה" hours: "2" minutes: "10" suffix: "בצהריים" } -> בשעה-14:10 בצהריים

    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour_to_noon = pynini.string_map([
            ("12", "12"),
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

        day_suffixes = (
            insert_space
            + pynutil.delete("suffix: \"")
            + (pynini.accep("בבוקר") | pynini.accep("לפנות בוקר"))
            + pynutil.delete("\"")
        )

        noon_suffixes = (
                insert_space
                + pynutil.delete("suffix: \"")
                + (pynini.accep("בצהריים") | pynini.accep("אחרי הצהריים") | pynini.accep("אחר הצהריים"))
                + pynutil.delete("\"")
        )

        evening_suffixes = (
                insert_space
                + pynutil.delete("suffix: \"")
                + (pynini.accep("בערב") | pynini.accep("לפנות ערב"))
                + pynutil.delete("\"")
        )

        night_suffixes = (
                insert_space
                + pynutil.delete("suffix: \"")
                + pynini.accep("בלילה")
                + pynutil.delete("\"")
        )

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

        prefix = (
            pynutil.delete("prefix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.insert("-")
            + pynutil.delete("\"")
        )

        optional_prefix = pynini.closure(prefix + delete_zero_or_one_space, 0, 1)
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

        graph |= optional_prefix + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
