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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time in Hebrew
        e.g. time { hours: "2" minutes: "55" suffix: "בלילה" } -> 2:55 בלילה
        e.g. time { hours: "2" minutes: "57" suffix: "בבוקר" } -> 2:57 בבוקר
        e.g. time { morphosyntactic_features: "ב" hours: "6" minutes: "32" suffix: "בערב" } -> ב-18:32 בערב
        e.g. time { morphosyntactic_features: "בשעה" hours: "2" minutes: "10" suffix: "בצהריים" } -> בשעה-14:10 בצהריים

    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour_to_noon = pynini.string_file(get_abs_path("data/time/hour_to_noon.tsv"))
        hour_to_evening = pynini.string_file(get_abs_path("data/time/hour_to_evening.tsv"))
        hour_to_night = pynini.string_file(get_abs_path("data/time/hour_to_night.tsv"))

        day_suffixes = pynini.string_file(get_abs_path("data/time/day_suffix.tsv"))
        day_suffixes = insert_space + pynutil.delete('suffix: "') + day_suffixes + pynutil.delete('"')

        noon_suffixes = pynini.string_file(get_abs_path("data/time/noon_suffix.tsv"))
        noon_suffixes = insert_space + pynutil.delete('suffix: "') + noon_suffixes + pynutil.delete('"')

        evening_suffixes = pynini.string_file(get_abs_path("data/time/evening_suffix.tsv"))
        evening_suffixes = insert_space + pynutil.delete('suffix: "') + evening_suffixes + pynutil.delete('"')

        night_suffixes = pynini.string_file(get_abs_path("data/time/night_suffix.tsv"))
        night_suffixes = insert_space + pynutil.delete('suffix: "') + night_suffixes + pynutil.delete('"')

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
        )

        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
        )

        prefix = (
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.insert("-")
            + pynutil.delete('"')
        )

        optional_prefix = pynini.closure(prefix + delete_zero_or_one_space, 0, 1)
        optional_suffix = pynini.closure(delete_space + day_suffixes, 0, 1)
        graph = hour + delete_space + pynutil.insert(":") + minute + optional_suffix

        for hour_to, suffix in zip(
            [hour_to_noon, hour_to_evening, hour_to_night],
            [noon_suffixes, evening_suffixes, night_suffixes],
        ):
            graph |= hour @ hour_to + delete_space + pynutil.insert(":") + minute + delete_space + suffix

        graph |= optional_prefix + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
