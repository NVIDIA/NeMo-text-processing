# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

time_zones = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))
suffix = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15 est" -> time { hours: "dos" minutes: "quince" zone: "e s t"}
        "2 h" -> time { hours: "dos" }
        "9 h" -> time { hours: "nueve" }
        "02:15:10 h" -> time { hours: "dos" minutes: "quince" seconds: "diez"}

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        delete_time_delimiter = pynutil.delete(pynini.union(".", ":"))

        fem_one = pynini.string_map([("un", "una"), ("ún", "una")])
        change_one = pynini.cdrewrite(fem_one, "", "", NEMO_SIGMA)
        cardinal_graph_fem = cardinal.graph @ change_one
        cardinal_graph_masc = cardinal.graph

        day_suffix = pynutil.insert("suffix: \"") + suffix + pynutil.insert("\"")
        day_suffix = delete_space + insert_space + day_suffix

        delete_hora_suffix = delete_space + insert_space + pynutil.delete("h")
        delete_minute_suffix = delete_space + insert_space + (pynutil.delete("min") | pynutil.delete("m"))
        delete_second_suffix = delete_space + insert_space + (pynutil.delete("seg") | pynutil.delete("s"))

        labels_hour_24 = [
            str(x) for x in range(0, 25)
        ]  # Can see both systems. Twelve hour requires am/pm for ambiguity resolution
        labels_hour_12 = [str(x) for x in range(1, 13)]
        labels_minute_or_second = [str(x) for x in range(1, 60)]

        delete_leading_zero_to_double_digit = (
            pynini.closure(pynutil.delete("0") | (NEMO_DIGIT - "0"), 0, 1) + NEMO_DIGIT
        )

        graph_24 = (
            pynini.closure(NEMO_DIGIT, 1, 2) @ delete_leading_zero_to_double_digit @ pynini.union(*labels_hour_24)
        )
        graph_12 = (
            pynini.closure(NEMO_DIGIT, 1, 2) @ delete_leading_zero_to_double_digit @ pynini.union(*labels_hour_12)
        )
        graph_minute_or_second = (
            pynini.closure(NEMO_DIGIT, 1, 2)
            @ delete_leading_zero_to_double_digit
            @ pynini.union(*labels_minute_or_second)
        )
        graph_hour_24 = graph_24 @ cardinal_graph_fem
        graph_hour_12 = graph_12 @ cardinal_graph_fem
        graph_minute_or_second @= cardinal_graph_masc

        final_graph_hour_only_24 = (
            pynutil.insert("hours: \"") + graph_hour_24 + pynutil.insert("\"") + delete_hora_suffix
        )
        final_graph_hour_only_12 = pynutil.insert("hours: \"") + graph_hour_12 + pynutil.insert("\"") + day_suffix

        final_graph_hour_24 = pynutil.insert("hours: \"") + graph_hour_24 + pynutil.insert("\"")
        final_graph_hour_12 = pynutil.insert("hours: \"") + graph_hour_12 + pynutil.insert("\"")

        final_graph_minute = pynutil.insert("minutes: \"") + graph_minute_or_second + pynutil.insert("\"")
        final_graph_second = pynutil.insert("seconds: \"") + graph_minute_or_second + pynutil.insert("\"")

        # handle suffixes like gmt or utc+1
        utc_one = pynini.string_map([("un", "uno"), ("ún", "uno")])
        change_utc_one = pynini.cdrewrite(utc_one, "", "", NEMO_SIGMA)
        utc_cardinal_graph = cardinal.graph @ change_utc_one

        utc_or_gmt_numbers = (NEMO_DIGIT @ utc_cardinal_graph) | (
            graph_hour_24 + delete_time_delimiter + pynutil.insert(" ") + graph_minute_or_second
        )

        utc_or_gmt_diff = (
            pynutil.delete(pynini.closure(" ", 0, 1))
            + pynutil.insert(" ")
            + pynini.string_map([("+", "más"), ("-", "menos")])
            + pynutil.delete(pynini.closure(" ", 0, 1))
            + pynutil.insert(" ")
            + utc_or_gmt_numbers
        )

        time_zone_graph = time_zones + pynini.closure(utc_or_gmt_diff, 0, 1)

        final_time_zone_optional = pynini.closure(
            delete_space + insert_space + pynutil.insert("zone: \"") + time_zone_graph + pynutil.insert("\""), 0, 1,
        )

        # 02.30 h
        graph_hm = (
            final_graph_hour_24
            + delete_time_delimiter
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + pynini.closure(
                (delete_time_delimiter + (pynutil.delete("00") | (insert_space + final_graph_second))), 0, 1
            )
            + pynini.closure(delete_hora_suffix, 0, 1)  # 2.30 is valid if unambiguous
            + final_time_zone_optional
        )

        # 2 h 30 min
        graph_hm |= (
            final_graph_hour_24
            + delete_hora_suffix
            + delete_space
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + delete_minute_suffix
            + pynini.closure(
                (delete_space + (pynutil.delete("00") | (insert_space + final_graph_second)) + delete_second_suffix),
                0,
                1,
            )
            + final_time_zone_optional
        )

        # 2.30 a. m. (Only for 12 hour clock)
        graph_hm |= (
            final_graph_hour_12
            + delete_time_delimiter
            + (pynutil.delete("00") | (insert_space + final_graph_minute))
            + pynini.closure(
                (delete_time_delimiter + (pynutil.delete("00") | (insert_space + final_graph_second))), 0, 1
            )
            + day_suffix
            + final_time_zone_optional
        )

        graph_h = (
            pynini.union(final_graph_hour_only_24, final_graph_hour_only_12) + final_time_zone_optional
        )  # Should always have a time indicator, else we'll pass to cardinals

        if not deterministic:
            # This includes alternate vocalization (hour menos min, min para hour), here we shift the times and indicate a `style` tag
            hour_shift_24 = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to_24.tsv")))
            hour_shift_12 = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to_12.tsv")))
            minute_shift = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))

            graph_hour_to_24 = graph_24 @ hour_shift_24 @ cardinal_graph_fem
            graph_hour_to_12 = graph_12 @ hour_shift_12 @ cardinal_graph_fem

            graph_minute_to = minute_shift @ graph_minute_or_second

            final_graph_hour_to_24 = pynutil.insert("hours: \"") + graph_hour_to_24 + pynutil.insert("\"")
            final_graph_hour_to_12 = pynutil.insert("hours: \"") + graph_hour_to_12 + pynutil.insert("\"")

            final_graph_minute_to = pynutil.insert("minutes: \"") + graph_minute_to + pynutil.insert("\"")

            graph_menos = pynutil.insert(" style: \"1\"")
            graph_para = pynutil.insert(" style: \"2\"")

            final_graph_style = graph_menos | graph_para

            # 02.30 h (omitting seconds since a bit awkward)
            graph_hm |= (
                final_graph_hour_to_24
                + delete_time_delimiter
                + insert_space
                + final_graph_minute_to
                + pynini.closure(delete_hora_suffix, 0, 1)  # 2.30 is valid if unambiguous
                + final_time_zone_optional
                + final_graph_style
            )

            # 2 h 30 min
            graph_hm |= (
                final_graph_hour_to_24
                + delete_hora_suffix
                + delete_space
                + insert_space
                + final_graph_minute_to
                + delete_minute_suffix
                + final_time_zone_optional
                + final_graph_style
            )

            # 2.30 a. m. (Only for 12 hour clock)
            graph_hm |= (
                final_graph_hour_to_12
                + delete_time_delimiter
                + insert_space
                + final_graph_minute_to
                + day_suffix
                + final_time_zone_optional
                + final_graph_style
            )

        final_graph = graph_hm | graph_h
        if deterministic:
            final_graph = final_graph + pynutil.insert(" preserve_order: true")
        final_graph = final_graph.optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
