# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
    Time formats that it converts:
    - <hour> + <minutes>
        e.g. la una diez -> time { hours: "la 1" minutes: "10" }
    - <hour> + " y " + <minutes>
        e.g. la una y diez -> time { hours: "la 1" minutes: "10" }
    - <hour> + " con " + <minutes>
        e.g. la una con diez -> time { hours: "la 1" minutes: "10" }
    - <hour> + " menos " + <minutes>
        e.g. las dos menos cuarto -> time { hours: "la 1" minutes: "45" }
    - "(un) cuarto para " + <hour>
        e.g. cuarto para las dos -> time { minutes: "45" hours: "la 1" }

    Note that times on the hour (e.g. "las dos" i.e. "two o'clock") do not get
    converted into a time format. This is to avoid converting phrases that are 
    not part of a time phrase (e.g. "las dos personas" i.e. "the two people")
        e.g. las dos -> tokens { name: "las" } tokens { name: "dos" }
    However, if a time on the hour is followed by a suffix (indicating 'a.m.' 
    or 'p.m.'), it will be converted.
        e.g. las dos pe eme -> time { hours: "las 2" minutes: "00" suffix: "p.m." }
    
    In the same way, times without a preceding article are not converted. This is 
    to avoid converting ranges or complex fractions
        e.g. dos y media -> tokens { name: "dos" } tokens { name: "y" } tokens { name: "media" }
    However, if a time without an article is followed by a suffix (indicating 'a.m.' 
    or 'p.m.'), it will be converted.
        e.g. dos y media p m -> time { hours: "2" minutes: "30" suffix: "p.m." }

    Note that although the TimeFst verbalizer can accept 'zone' (timezone) fields, 
    so far the rules have not been added to the TimeFst tagger to process
    timezones (to keep the rules simple, and because timezones are not very
    often specified in Spanish.)
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        suffix_graph = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))
        time_to_graph = pynini.string_file(get_abs_path("data/time/time_to.tsv"))
        minutes_to_graph = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))
        time_zones = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))
        time_zones = pynini.invert(time_zones)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        graph_1_to_100 = pynini.union(
            graph_digit,
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + pynutil.delete(" y ") + graph_digit),
        )

        # note that graph_hour will start from 2 hours
        # "1 o'clock" will be treated differently because it
        # is singular
        digits_2_to_23 = [str(digits) for digits in range(2, 24)]
        digits_1_to_59 = [str(digits) for digits in range(1, 60)]

        graph_1oclock = pynini.cross("la una", "la 1")
        graph_hour = pynini.cross("las ", "las ") + graph_1_to_100 @ pynini.union(*digits_2_to_23)
        graph_minute = graph_1_to_100 @ pynini.union(*digits_1_to_59)
        graph_minute_verbose = pynini.cross("media", "30") | pynini.cross("cuarto", "15")

        final_graph_hour = pynutil.insert("hours: \"") + (graph_1oclock | graph_hour) + pynutil.insert("\"")

        final_graph_minute = (
            pynutil.insert("minutes: \"")
            + pynini.closure((pynutil.delete("y") | pynutil.delete("con")) + delete_space, 0, 1)
            + (graph_minute | graph_minute_verbose)
            + pynutil.insert("\"")
        )

        # g m t más tres -> las 2:00 p.m. gmt+3
        digits_1_to_23 = [str(digits) for digits in range(1, 24)]
        offset = graph_1_to_100 @ pynini.union(*digits_1_to_23)
        sign = pynini.cross("más", "+") | pynini.cross("menos", "-")
        full_offset = pynutil.delete(" ") + sign + pynutil.delete(" ") + offset
        graph_offset = pynini.closure(full_offset, 0, 1)
        graph_time_zones = pynini.accep(" ") + time_zones + graph_offset
        time_zones_optional = pynini.closure(graph_time_zones, 0, 1)

        final_suffix = pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\"")
        final_suffix_optional = pynini.closure(delete_space + insert_space + final_suffix, 0, 1)

        final_time_zone_optional = pynini.closure(
            delete_space
            + insert_space
            + pynutil.insert("zone: \"")
            + convert_space(time_zones_optional)
            + pynutil.insert("\""),
            0,
            1,
        )

        # las nueve a eme (only convert on-the-hour times if they are followed by a suffix)
        graph_1oclock_with_suffix = pynini.closure(pynini.accep("la "), 0, 1) + pynini.cross("una", "1")
        graph_hour_with_suffix = pynini.closure(pynini.accep("las "), 0, 1) + graph_1_to_100 @ pynini.union(
            *digits_2_to_23
        )
        final_graph_hour_with_suffix = (
            pynutil.insert("hours: \"") + (graph_1oclock_with_suffix | graph_hour_with_suffix) + pynutil.insert("\"")
        )

        graph_hsuffix = (
            final_graph_hour_with_suffix
            + delete_extra_space
            + pynutil.insert("minutes: \"00\"")
            + insert_space
            + final_suffix
            + final_time_zone_optional
        )

        # las nueve y veinticinco
        graph_hm = final_graph_hour + delete_extra_space + final_graph_minute

        # nueve y veinticinco a m
        graph_hm_suffix = (
            final_graph_hour_with_suffix + delete_extra_space + final_graph_minute + delete_extra_space + final_suffix
        )

        # un cuarto para las cinco
        graph_mh = (
            pynutil.insert("minutes: \"")
            + minutes_to_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + pynutil.insert("hours: \"")
            + time_to_graph
            + pynutil.insert("\"")
        )

        # las diez menos diez
        graph_time_to = (
            pynutil.insert("hours: \"")
            + time_to_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + pynutil.insert("minutes: \"")
            + delete_space
            + pynutil.delete("menos")
            + delete_space
            + pynini.union(
                pynini.cross("cinco", "55"),
                pynini.cross("diez", "50"),
                pynini.cross("cuarto", "45"),
                pynini.cross("veinte", "40"),
                pynini.cross("veinticinco", "30"),
            )
            + pynutil.insert("\"")
        )

        final_graph = pynini.union(
            (graph_hm | graph_mh | graph_time_to) + final_suffix_optional, graph_hsuffix, graph_hm_suffix
        ).optimize()

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
