# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """Classifies Polish numeric hours and minutes."""

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        hour_numbers = pynini.union(*(str(hour) for hour in range(1, 24)))
        hours = hour_numbers | pynutil.delete("0") + pynini.union(*(str(hour) for hour in range(1, 10)))
        minutes = pynini.union(*(f"{minute:02d}" for minute in range(1, 60)))
        minute_words = (pynutil.delete("0") + cardinal.graphs["mi_sg_nom"]) | cardinal.graphs["mi_sg_nom"]

        def time_graph(hour_slot: str, prefix: 'pynini.FstLike') -> 'pynini.FstLike':
            hour = hours @ ordinal.graphs[hour_slot]
            hour_field = pynutil.insert('hours: "') + prefix + hour + pynutil.insert('"')
            minute_field = pynutil.insert(' minutes: "') + (minutes @ minute_words) + pynutil.insert('"')
            separator = pynutil.delete(pynini.union(":", "."))
            return hour_field + separator + (pynutil.delete("00") | minute_field)

        plain = time_graph("f_sg_nom", pynini.accep(""))
        governed = time_graph("f_sg_loc", pynini.accep("o") + delete_space + pynutil.insert(" "))
        hour_abbreviation = (
            pynini.accep("o")
            + delete_space
            + pynutil.insert(" ")
            + pynini.cross("godz.", "godzinie")
            + delete_space
            + pynutil.insert(" ")
        )
        governed |= time_graph("f_sg_loc", hour_abbreviation)

        locale_hour = pynini.cross("00", "zero") | hours @ ordinal.graphs["f_sg_nom"]
        locale_minute = pynini.cross("00", "zero") | minutes @ minute_words
        locale_second = pynini.cross("00", "zero") | minutes @ minute_words
        locale_time = (
            pynutil.insert('hours: "')
            + locale_hour
            + pynutil.insert('"')
            + pynutil.delete(":")
            + pynutil.insert(' minutes: "')
            + locale_minute
            + pynutil.insert('"')
            + pynutil.delete(":")
            + pynutil.insert(' seconds: "')
            + locale_second
            + pynutil.insert('"')
        )
        self.alternative_graph = pynini.Fst()
        if not deterministic:
            alternatives = []
            hour_to = load_labels(get_abs_path("data/time/hour_to.tsv"))
            minute_to_half = load_labels(get_abs_path("data/time/minute_to_half.tsv"))
            minute_from_half = load_labels(get_abs_path("data/time/minute_from_half.tsv"))
            minute_phrases = minute_to_half + [["30", "wpół do"]] + minute_from_half
            for hour, following in hour_to:
                for minute, phrase in minute_phrases:
                    spoken = f"{phrase} {following}"
                    for separator in (":", "."):
                        alternatives.append((f"{hour}{separator}{minute}", spoken))
                        alternatives.append((f"{int(hour):02d}{separator}{minute}", spoken))
            self.alternative_graph = (
                pynutil.insert('hours: "') + pynini.string_map(alternatives) + pynutil.insert('"')
            ).optimize()

        self.final_graph = (plain | governed | locale_time | self.alternative_graph).optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
