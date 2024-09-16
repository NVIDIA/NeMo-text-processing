# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.sv.graph_utils import ensure_space
from nemo_text_processing.text_normalization.sv.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        12:30 e.m. est -> time { hours: "tolv" minutes: "trettio" suffix: "eftermiddag" zone: "e s t" }
        2.30 e.m. -> time { hours: "två" minutes: "trettio" suffix: "eftermiddag" }
        02.30 e.m. -> time { hours: "två" minutes: "trettio" suffix: "eftermiddag" }
        2.00 e.m. -> time { hours: "två" suffix: "eftermiddag" }
        kl. 2 e.m. -> time { hours: "två" suffix: "eftermiddag" }
        02:00 -> time { hours: "två" }
        2:00 -> time { hours: "två" }
        10:00:05 e.m. -> time { hours: "tio" minutes: "noll" seconds: "fem" suffix: "eftermiddag" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        suffix_graph = pynini.string_map(load_labels(get_abs_path("data/time/suffix.tsv")))
        time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = cardinal.graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        )

        time_sep = pynutil.delete(pynini.union(":", "."))
        klockan = pynini.union(pynini.cross("kl.", "klockan"), pynini.cross("kl", "klockan"), "klockan", "klockan är")
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
                pynutil.insert("minutes: \"") + pynini.cross("00", "noll noll") + pynutil.insert("\"")
            )
        final_graph_second = (
            pynutil.insert("seconds: \"")
            + (pynutil.delete("0") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        if not deterministic:
            final_graph_second |= (
                pynutil.insert("seconds: \"")
                + (pynini.cross("0", "noll") + insert_space + graph_minute_single | graph_minute_double)
                + pynutil.insert("\"")
            )
            final_graph_second |= (
                pynutil.insert("seconds: \"") + pynini.cross("00", "noll noll") + pynutil.insert("\"")
            )
        final_suffix = pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\"")
        final_suffix_optional = pynini.closure(ensure_space + final_suffix, 0, 1)
        final_time_zone = pynutil.insert("zone: \"") + convert_space(time_zone_graph) + pynutil.insert("\"")
        final_time_zone_optional = pynini.closure(NEMO_SPACE + final_time_zone, 0, 1,)

        # 2:30 pm, 02:30, 2:00
        graph_hm_kl = (
            klockan_hour_graph
            + time_sep
            + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
            + final_suffix_optional
            + final_time_zone_optional
        )
        graph_hm_sfx = (
            final_graph_hour
            + time_sep
            + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
            + ensure_space
            + (final_suffix + final_time_zone_optional | final_time_zone)
        )
        graph_hm = graph_hm_kl | graph_hm_sfx

        # 10:30:05 pm,
        graph_hms_sfx = (
            final_graph_hour
            + time_sep
            + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
            + time_sep
            + (pynini.cross("00", " seconds: \"noll\"") | insert_space + final_graph_second)
            + ensure_space
            + (final_suffix + final_time_zone_optional | final_time_zone)
        )
        graph_hms_sfx |= (
            final_graph_hour
            + pynutil.delete(".")
            + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
            + pynutil.delete(".")
            + (pynini.cross("00", " seconds: \"noll\"") | insert_space + final_graph_second)
            + ensure_space
            + (final_suffix + final_time_zone_optional | final_time_zone)
        )
        graph_hms_kl = (
            klockan_hour_graph
            + pynutil.delete(":")
            + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
            + pynutil.delete(":")
            + (pynini.cross("00", " seconds: \"noll\"") | insert_space + final_graph_second)
            + final_suffix_optional
            + final_time_zone_optional
        )
        graph_hms_kl |= (
            klockan_hour_graph
            + pynutil.delete(".")
            + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
            + pynutil.delete(".")
            + (pynini.cross("00", " seconds: \"noll\"") | insert_space + final_graph_second)
            + final_suffix_optional
            + final_time_zone_optional
        )
        graph_hms = graph_hms_kl | graph_hms_sfx
        if not deterministic:
            graph_hms |= (
                final_graph_hour
                + pynutil.delete(".")
                + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
                + pynutil.delete(".")
                + (pynini.cross("00", " seconds: \"noll\"") | insert_space + final_graph_second)
            )
            graph_hms |= (
                final_graph_hour
                + pynutil.delete(":")
                + (pynini.cross("00", " minutes: \"noll\"") | insert_space + final_graph_minute)
                + pynutil.delete(":")
                + (pynini.cross("00", " seconds: \"noll\"") | insert_space + final_graph_second)
            )
        self.graph_hms = graph_hms
        self.graph_hm = graph_hm
        # 2 pm est
        ins_minutes = pynutil.insert(" minutes: \"noll\"")
        graph_h = (
            final_graph_hour + ins_minutes + ensure_space + (final_suffix + final_time_zone_optional | final_time_zone)
        )
        graph_h |= klockan_hour_graph + ins_minutes + final_suffix_optional + final_time_zone_optional
        self.graph_h = graph_h

        final_graph = (graph_hm | graph_h | graph_hms).optimize() @ pynini.cdrewrite(
            delete_extra_space, "", "", NEMO_SIGMA
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
