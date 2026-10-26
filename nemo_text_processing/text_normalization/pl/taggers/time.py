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
from nemo_text_processing.text_normalization.pl.inflection import inflect_noun
from nemo_text_processing.text_normalization.pl.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """Classifies Polish numeric hours and minutes."""

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        hour_numbers = pynini.union(*(str(hour) for hour in range(1, 24)))
        hours = hour_numbers | pynutil.delete("0") + pynini.union(*(str(hour) for hour in range(1, 10)))
        minutes = pynini.union(*(f"{minute:02d}" for minute in range(1, 60)))
        minute_words = (pynutil.delete("0") + cardinal.graphs["mi_sg_nom"]) | cardinal.graphs["mi_sg_nom"]
        time_zone = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))
        time_zone_suffix = delete_space + pynutil.insert(' zone: "') + time_zone + pynutil.insert('"')
        optional_time_zone = pynini.closure(time_zone_suffix, 0, 1)

        def time_graph(hour_slot: str, prefix: 'pynini.FstLike') -> 'pynini.FstLike':
            hour = hours @ ordinal.graphs[hour_slot]
            hour_value = prefix + hour
            hour_field = pynutil.insert('hours: "') + hour_value + pynutil.insert('"')
            hour_with_noun_field = pynutil.insert('hours: "') + hour_value + pynutil.insert(' godzina"')
            minute_field = pynutil.insert(' minutes: "') + (minutes @ minute_words) + pynutil.insert('"')
            separator = pynutil.delete(pynini.union(":", "."))
            minute_value = pynutil.delete("00") | minute_field
            graph = hour_field + separator + minute_value + optional_time_zone
            if not deterministic:
                graph |= hour_with_noun_field + separator + minute_value + optional_time_zone
            return graph

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
        second_lemma, second_gender, second_grammar = next(
            fields[1:] for fields in load_labels(get_abs_path("data/measures/units.tsv")) if fields[0] == "s"
        )
        second_forms = inflect_noun(second_lemma, second_grammar)
        second_one = pynini.cross("01", "1") @ cardinal.graphs[f"{second_gender}_sg_nom"]
        few_seconds = [second for second in range(1, 60) if second % 10 in {2, 3, 4} and second not in {12, 13, 14}]
        many_seconds = [second for second in range(1, 60) if second != 1 and second not in few_seconds]
        second_few = (
            pynini.union(*(pynini.cross(f"{second:02d}", str(second)) for second in few_seconds))
            @ cardinal.graphs[f"{second_gender}_pl_nom"]
        )
        second_many = (
            pynini.union(*(pynini.cross(f"{second:02d}", str(second)) for second in many_seconds))
            @ cardinal.graphs[f"{second_gender}_pl_nom"]
        )
        second_values = (
            second_one + pynutil.insert(" " + second_forms["sg_nom"])
            | second_few + pynutil.insert(" " + second_forms["pl_nom"])
            | second_many + pynutil.insert(" " + second_forms["pl_gen"])
        )
        locale_second = second_values
        locale_time = (
            pynutil.insert('hours: "')
            + locale_hour
            + pynutil.insert('"')
            + pynutil.delete(":")
            + (pynutil.delete("00") | pynutil.insert(' minutes: "') + locale_minute + pynutil.insert('"'))
            + pynutil.delete(":")
            + (pynutil.delete("00") | pynutil.insert(' seconds: "') + locale_second + pynutil.insert('"'))
            + optional_time_zone
        )
        duration_units = {
            fields[0]: (fields[1], fields[2], inflect_noun(fields[1], fields[3]))
            for fields in load_labels(get_abs_path("data/measures/units.tsv"))
            if fields[0] in {"godz", "min", "s"}
        }

        def duration_number(unit, numbers):
            lemma, gender, forms = duration_units[unit]
            few_numbers = [number for number in numbers if number % 10 in {2, 3, 4} and number not in {12, 13, 14}]
            many_numbers = [number for number in numbers if number != 1 and number not in few_numbers]
            one = pynini.cross("01", "1") @ cardinal.graphs[f"{gender}_sg_nom"]
            few = (
                pynini.union(*(pynini.cross(f"{number:02d}", str(number)) for number in few_numbers))
                @ cardinal.graphs[f"{gender}_pl_nom"]
            )
            many = (
                pynini.union(*(pynini.cross(f"{number:02d}", str(number)) for number in many_numbers))
                @ cardinal.graphs[f"{gender}_pl_nom"]
            )
            return (
                one + pynutil.insert(" " + forms["sg_nom"])
                | few + pynutil.insert(" " + forms["pl_nom"])
                | many + pynutil.insert(" " + forms["pl_gen"])
            )

        duration_hours = duration_number("godz", range(1, 24))
        duration_minutes = duration_number("min", range(1, 60))
        duration_seconds = duration_number("s", range(1, 60))
        duration_time = (
            pynutil.insert('duration: "true"')
            + (pynutil.delete("00") | pynutil.insert(' hours: "') + duration_hours + pynutil.insert('"'))
            + pynutil.delete(":")
            + (pynutil.delete("00") | pynutil.insert(' minutes: "') + duration_minutes + pynutil.insert('"'))
            + pynutil.delete(":")
            + (pynutil.delete("00") | pynutil.insert(' seconds: "') + duration_seconds + pynutil.insert('"'))
        )
        until_half = []
        hour_to_nom = load_labels(get_abs_path("data/time/hour_to_nom.tsv"))
        for hour, following in hour_to_nom:
            for minute in range(31, 60):
                remaining = 60 - minute
                remaining_graph = pynini.cross(f"{minute:02d}", str(remaining)) @ cardinal.graphs["mi_sg_nom"]
                hour_graph = pynini.cross(hour, "za ") + pynutil.delete(":") + remaining_graph
                if int(hour) < 10:
                    hour_graph |= pynini.cross(f"{int(hour):02d}", "za ") + pynutil.delete(":") + remaining_graph
                until_half.append(hour_graph + pynutil.insert(" " + following))
        until_half_graph = (
            pynutil.insert('hours: "') + pynini.union(*until_half) + pynutil.insert('"') + optional_time_zone
        )
        locale_time |= until_half_graph
        legacy_second = pynini.cross("00", "zero") | (minutes @ minute_words)
        legacy_time = (
            pynutil.insert('legacy: "true" hours: "')
            + locale_hour
            + pynutil.delete(":")
            + pynutil.insert('" minutes: "')
            + locale_minute
            + pynutil.delete(":")
            + pynutil.insert('" seconds: "')
            + legacy_second
            + pynutil.insert('"')
        )
        special = pynini.string_file(get_abs_path("data/time/special.tsv"))
        if deterministic:
            locale_time = (
                pynutil.add_weight(special, -0.002)
                | pynutil.add_weight(until_half_graph, -0.001)
                | pynutil.add_weight(duration_time, 0.001)
            )
        else:
            locale_time |= special | duration_time | legacy_time
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
