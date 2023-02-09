# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan.
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
    NEMO_SPACE,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hu.utils import get_abs_path, naive_inflector
from pynini.lib import pynutil


QUARTERS = {
    15: "negyed",
    30: "fél",
    45: "háromnegyed"
}
def get_all_to_or_from_numbers():
    output = {}
    for num, word in QUARTERS.items():
        current_past = []
        current_to = []
        for i in range(1, 60):
            if i == num:
                continue
            elif i < num:
                current_to.append((str(i), str(num-i)))
            else:
                current_past.append((str(i), str(i-num)))
        output[word] = {}
        output[word]["past"] = current_past
        output[word]["to"] = current_to
    return output


def get_all_to_or_from_fst(cardinal: GraphFst):
    numbers = get_all_to_or_from_numbers()
    output = {}
    for key in numbers:
        for when in ["past", "to"]:
            output[key] = {}
            map = pynini.string_map(numbers[key][when])
            output[key][when] = pynini.project(map, "input") @ map @ cardinal.graph
    return output


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "Délelőtt 9 óra est" -> time { hours: "2" minutes: "15" zone: "e s t"}
        "9 óra" -> time { hours: "2" }
        "09:00 óra" -> time { hours: "2" }
        "02:15:10 óra" -> time { hours: "2" minutes: "15" seconds: "10"}
        "negyed 2" -> time { minutes: "15" hours: "1" }
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        ora_word = pynini.cross("ó", "óra") | pynini.accep("óra")
        ora_suffix = pynutil.delete(" ") + pynutil.delete(ora_word)
        ora_forms = pynini.string_map(naive_inflector("ó", "óra", True) + [("ó", "óra")])
        perc_forms = pynini.string_map(naive_inflector("p", "perc", True) + [("p", "perc")])
        masodperc_forms = pynini.string_map(naive_inflector("mp", "másodperc", True) + [("mp", "másodperc")])
        final_forms = ora_forms | perc_forms | masodperc_forms
        final_forms |= pynini.project(final_forms, "output")
        final_suffix = pynutil.insert("suffix: \"") + final_forms + pynutil.insert("\"")
        time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))

        labels_hour = [str(x) for x in range(0, 25)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        minutes_to = pynini.string_map([(str(i), str(60-i)) for i in range(1, 60)])
        minutes_inverse = pynini.invert(
            pynini.project(minutes_to, "input")
            @ cardinal.graph
        )
        self.minute_words_to_words = minutes_inverse @ minutes_to @ cardinal.graph
        minute_words_to_words = pynutil.insert("minutes: \"") + self.minute_words_to_words + pynutil.insert("\"")
        def hours_to_pairs():
            for x in range(1, 13):
                if x == 12:
                    y = 1
                else:
                    y = x + 1
                yield y, x
        hours_next = pynini.string_map([(str(x[0]), str(x[1])) for x in hours_to_pairs()])
        hours_next_inverse = pynini.invert(
            pynini.project(hours_next, "input")
            @ cardinal.graph
        )
        self.hour_numbers_to_words = hours_next @ cardinal.graph
        self.hour_words_to_words = hours_next_inverse @ self.hour_numbers_to_words
        hour_numbers_to_words = pynutil.insert("hours: \"") + self.hour_numbers_to_words + pynutil.insert("\"")
        hour_words_to_words = pynutil.insert("hours: \"") + self.hour_words_to_words + pynutil.insert("\"")

        quarter_map = pynini.string_map([(p[1], str(p[0])) for p in QUARTERS.items()])
        quarter_map_graph = pynutil.insert("minutes: \"") + (quarter_map @ cardinal.graph) + pynutil.insert("\"")
        quarter_words = pynini.string_map(QUARTERS.values())
        quarter_words_graph = pynutil.insert("minutes: \"") + quarter_words + pynutil.insert("\"")
        # {quarter} {hour_next}
        # negyed 2 -> minutes: "tizenöt" hours: "egy"
        self.quarter_prefixed_next_to_current = quarter_map_graph + NEMO_SPACE + hour_numbers_to_words
        # For ITN
        self.quarter_prefixed_next_to_current_words = quarter_map_graph + NEMO_SPACE + hour_words_to_words

        delete_leading_zero_to_double_digit = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT

        graph_hour = pynini.union(*labels_hour)

        graph_minute_single = pynini.union(*labels_minute_single)
        graph_minute_double = pynini.union(*labels_minute_double)

        final_graph_hour_only = pynutil.insert("hours: \"") + graph_hour + pynutil.insert("\"")
        final_graph_hour = (
            pynutil.insert("hours: \"") + delete_leading_zero_to_double_digit @ graph_hour + pynutil.insert("\"")
        )
        hour_only_delimited = (
            pynutil.insert("hours: \"")
            + delete_leading_zero_to_double_digit @ graph_hour
            + NEMO_SPACE
            + ora_forms
            + pynutil.insert("\"")
            + pynutil.insert(" preserve_order: true")
        )
        final_graph_minute = (
            pynutil.insert("minutes: \"")
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        final_graph_second = (
            pynutil.insert("seconds: \"")
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        final_time_zone_optional = pynini.closure(
            pynini.accep(" ") + pynutil.insert("zone: \"") + convert_space(time_zone_graph) + pynutil.insert("\""),
            0,
            1,
        )

        # 02:30 Uhr
        graph_hm = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + final_suffix
            + final_time_zone_optional
        )

        # 10:30:05 Uhr,
        graph_hms = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", " minutes: \"0\"") | (insert_space + final_graph_minute))
            + pynutil.delete(":")
            + (pynini.cross("00", " seconds: \"0\"") | (insert_space + final_graph_second))
            + final_suffix
            + final_time_zone_optional
            + pynutil.insert(" preserve_order: true")
        )

        # 2 Uhr est
        graph_h = final_graph_hour_only + final_suffix + final_time_zone_optional
        word_delim = hour_only_delimited
        final_graph = (graph_hm | graph_h | graph_hms | word_delim).optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
