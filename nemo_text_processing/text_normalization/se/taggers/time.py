# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    insert_space,
)
from nemo_text_processing.text_normalization.se.graph_utils import ensure_space
from nemo_text_processing.text_normalization.se.utils import get_abs_path, load_labels

QUARTERS = {15: "kvárta badjel", 30: "beal", 45: "kvárta váile"}


def get_all_to_or_from_numbers():
    output = {}
    for num, word in QUARTERS.items():
        current_past = []
        current_to = []
        for i in range(1, 60):
            if i == num:
                continue
            elif i < num:
                current_to.append((str(i), str(num - i)))
            else:
                current_past.append((str(i), str(i - num)))
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
        13:30 cst -> time { hours: "golbmanuppelohkái" minutes: "golbmalogi" zone: "c s t" }
        dii. 13.30 -> time { hours: "golbmanuppelohkái" minutes: "golbmalogi" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))
        klockan = pynini.string_file(get_abs_path("data/time/prefix.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = cardinal.graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        )

        time_sep = pynutil.delete(pynini.union(":", "."))
        klockan_graph_piece = pynutil.insert("hours: \"") + klockan

        graph_hour = delete_leading_zero_to_double_digit @ pynini.union(*labels_hour) @ cardinal

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal
        klockan_hour_graph = klockan_graph_piece + ensure_space + graph_hour + pynutil.insert("\"")

        final_graph_hour = pynutil.insert("hours: \"") + graph_hour + pynutil.insert("\"")
        final_graph_minute = (
            pynutil.insert("minutes: \"")
            + (pynutil.delete("0") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        if not deterministic:
            final_graph_minute |= (
                pynutil.insert("minutes: \"")
                + (pynutil.delete("0") + insert_space + graph_minute_single | graph_minute_double)
                + pynutil.insert("\"")
            )
            final_graph_minute |= (
                pynutil.insert("minutes: \"") + pynini.cross("00", "nolla nolla") + pynutil.insert("\"")
            )
        final_graph_second = (
            pynutil.insert("seconds: \"")
            + (pynutil.delete("0") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        if not deterministic:
            final_graph_second |= (
                pynutil.insert("seconds: \"")
                + (pynini.cross("0", "nolla") + insert_space + graph_minute_single | graph_minute_double)
                + pynutil.insert("\"")
            )
            final_graph_second |= (
                pynutil.insert("seconds: \"") + pynini.cross("00", "nolla nolla") + pynutil.insert("\"")
            )
        final_time_zone = pynutil.insert("zone: \"") + convert_space(time_zone_graph) + pynutil.insert("\"")
        final_time_zone_optional = pynini.closure(
            NEMO_SPACE + final_time_zone,
            0,
            1,
        )

        hour = klockan_hour_graph | final_graph_hour
        minute = pynini.cross("00", " minutes: \"nolla\"") | insert_space + final_graph_minute
        second = pynini.cross("00", " seconds: \"nolla\"") | insert_space + final_graph_second

        graph_hm = hour + time_sep + minute + final_time_zone_optional
        graph_hms = hour + time_sep + minute + time_sep + second + final_time_zone_optional
        self.graph_hms = graph_hms
        self.graph_hm = graph_hm

        ins_minutes = pynutil.insert(" minutes: \"nolla\"")
        graph_h = klockan_hour_graph + ins_minutes + final_time_zone_optional
        graph_h |= final_graph_hour + ins_minutes + NEMO_SPACE + final_time_zone
        self.graph_h = graph_h

        final_graph = (graph_hm | graph_h | graph_hms).optimize() @ pynini.cdrewrite(
            delete_extra_space, "", "", NEMO_SIGMA
        )
        zero_fields = pynini.string_map(
            [
                ('minutes: "nolla"', 'minutes: "nolla nolla"'),
                ('seconds: "nolla"', 'seconds: "nolla nolla"'),
            ]
        )
        final_graph @= pynini.cdrewrite(zero_fields, "", "", NEMO_SIGMA)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
