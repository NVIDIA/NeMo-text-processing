# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_SPACE,
    NEMO_ALPHA,
    NEMO_SIGMA,
    TO_UPPER,
    GraphFst,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. acht uhr e s t-> time { hours: "8" minutes: "00" zone: "est" }
        e.g. dreizehn uhr -> time { hours: "13" minutes: "00 }
        e.g. dreizehn uhr zehn -> time { hours: "13" minutes: "10" }
        e.g. Viertel vor zwölf -> time { minutes: "45" hours: "11" }
        e.g. Viertel nach zwölf -> time { minutes: "15" hours: "12" }
        e.g. halb zwölf -> time { minutes: "30" hours: "11" }
        e.g. drei vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei nach zwölf -> time { minutes: "3" hours: "12" }
        e.g. drei uhr zehn minuten zehn sekunden -> time { hours: "3" hours: "10" seconds: "10"}
        e.g. Viertel sieben -> time { minutes: "15" hours: "6"}
        e.g. drei Viertel sieben -> time { minutes: "45" hours: "6"}
        e.g. ab halb acht abends -> time { suffix: "ab" hour: "7" minutes: "30" suffix: "abends" }
        e.g. zwischen dreizehn Uhr und fünfzehn Uhr -> time { suffix: "zwischen" hour: "13" minutes: "00" suffix: "und" hour: "15" minutes: "00"}
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")
        graph_integers_1_99 = cardinal.graph_single_and_double_digits
        graph_single_digits = cardinal.digits
        graph_dozen = cardinal.dozen
        graph_zero = pynini.string_map([("null", "00")])
        delete_hour = pynutil.delete("Uhr") | pynutil.delete("uhr")
        remove_und = pynutil.delete("und")
        hours_downshifted = pynini.string_file(
            get_abs_path("data/time/hours_downshifted.tsv")
        )

        hours = graph_integers_1_99 | graph_zero
        # Since minutes and seconds are expressed as double-digit numbers with leading zeros, the same WFST will be used for both
        minutes_or_seconds = (
            pynutil.add_weight((pynutil.insert("0") + graph_single_digits), -0.01)
            | graph_integers_1_99
            | graph_zero
        )
        minutes_units = (
            pynutil.delete("Minute")
            | pynutil.delete("Minuten")
            | pynutil.delete("minute")
            | pynutil.delete("minuten")
        )
        seconds_units = (
            pynutil.delete("Sekunde")
            | pynutil.delete("Sekunden")
            | pynutil.delete("sekunde")
            | pynutil.delete("sekunden")
        )

        # Formal time expressions (e.g. 10:30 -> zehn Uhr dreißig)
        graph_hour_only = (
            pynutil.insert("hours:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + hours
            + pynutil.insert('"')
            + pynutil.delete(NEMO_SPACE)
            + delete_hour
        )

        graph_hour_only_no_Uhr = (
            pynutil.insert("hours:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + hours
            + pynutil.insert('"')
        )

        graph_minutes_only = (
            pynutil.insert("minutes:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + minutes_or_seconds
            + pynutil.insert('"')
            + (
                pynutil.delete(NEMO_SPACE) + minutes_units
            ).ques  # The "Minute/-n" time unit can be omitted
        )

        graph_seconds_only = (
            pynutil.insert("seconds:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + minutes_or_seconds
            + pynutil.insert('"')
            + (pynutil.delete(NEMO_SPACE) + seconds_units).ques
        )

        graph_hour_and_minutes = (
            graph_hour_only + pynini.accep(NEMO_SPACE) + graph_minutes_only
        )

        graph_hour_and_minutes_no_Uhr = (
            graph_hour_only_no_Uhr + pynini.accep(NEMO_SPACE) + graph_minutes_only
        )

        graph_hour_and_seconds = (
            graph_hour_only + pynini.accep(NEMO_SPACE) + graph_seconds_only
        )

        graph_hour_and_seconds_no_Uhr = (
            graph_hour_only_no_Uhr + pynini.accep(NEMO_SPACE) + graph_seconds_only
        )

        graph_hour_and_minutes_and_seconds = (
            graph_hour_only
            + pynini.accep(NEMO_SPACE)
            + graph_minutes_only
            + pynini.accep(NEMO_SPACE)
            + (remove_und + pynutil.delete(NEMO_SPACE)).ques
            + graph_seconds_only
        )

        graph_hour_and_minutes_and_seconds_no_Uhr = (
            graph_hour_only_no_Uhr
            + pynini.accep(NEMO_SPACE)
            + graph_minutes_only
            + pynini.accep(NEMO_SPACE)
            + graph_seconds_only
        )

        # The combinations missing the hourly exponent (e.g. 24 Minuten 34 Sekunden) don't qualify as time expressions
        # They are instead included in the MEASURE class as examples of time units

        graph_time_formal = (
            graph_hour_only
            | graph_hour_and_minutes
            | pynutil.add_weight(graph_hour_and_seconds, 0.01)
            | graph_hour_and_minutes_and_seconds
        )

        # The graph below will be implemented in prepositional and range expressions where "Uhr" is frequenlyt left out

        graph_time_formal_no_unit = (
            graph_hour_only_no_Uhr
            | graph_hour_and_minutes_no_Uhr
            | pynutil.add_weight(graph_hour_and_seconds_no_Uhr, 0.01)
            | graph_hour_and_minutes_and_seconds_no_Uhr
        )

        # Informal time expressions (e.g. halb elf morgens -> 10:30)

        postpositions = pynini.string_file(get_abs_path("data/time/postpositions.tsv"))
        min_to_hours = pynini.string_file(get_abs_path("data/time/min_to_hour.tsv"))
        quarter = pynini.cross("Viertel", "15") | pynini.cross("viertel", "15")
        three_quarters = (
            pynini.cross("drei Viertel", "45")
            | pynini.cross("drei viertel", "45")
            | pynini.cross("dreiviertel", "45")
        )

        graph_postpositions = (
            pynutil.insert("suffix:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + postpositions
            + pynutil.insert('"')
        )

        graph_half_hour = (
            pynutil.delete("halb")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.insert("hours:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + hours_downshifted
            + pynutil.insert('"')
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('minutes: "30"')
        )

        graph_min_or_sec_past_hour = (
            pynutil.insert("minutes:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + (minutes_or_seconds | quarter | three_quarters)
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + ((minutes_units | seconds_units) + pynutil.delete(NEMO_SPACE)).ques
            + pynutil.delete("nach")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.insert("hours:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + hours
            + pynutil.insert('"')
            + (pynutil.delete(NEMO_SPACE) + delete_hour).ques
        )

        graph_min_or_sec_to_hour = (
            pynutil.insert("minutes:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + ((minutes_or_seconds | quarter) @ min_to_hours)
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + ((minutes_units | seconds_units) + pynutil.delete(NEMO_SPACE)).ques
            + pynutil.delete("vor")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.insert("hours:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + hours_downshifted
            + pynutil.insert('"')
            + (pynutil.delete(NEMO_SPACE) + delete_hour).ques
        )

        # Colloquial time expressions common across the dialects
        # The preposition "um" is commonly used in the dialects spoken in the Eastern part of Germany
        # to indicate full hours (e.g. um sechs -> 6).
        # It is commented out for the standard German varieties.

        # graph_um_full_hour = (
        #     pynutil.delete("um")
        #     + pynutil.delete(NEMO_SPACE)
        #     + pynutil.insert("hours:")
        #     + pynutil.insert(NEMO_SPACE)
        #     + pynutil.insert('"')
        #     + hours
        #     + pynutil.insert('"')
        #     + pynutil.delete(NEMO_SPACE)
        #     + delete_hour
        # )

        # Quarter hours (e.g. Viertel sieben -> 6:15, drei Viertel sieben -> 6:45)

        graph_quarter_hours = (
            pynutil.insert("minutes:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + (quarter | three_quarters)
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + pynutil.insert("hours:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + hours_downshifted
            + pynutil.insert('"')
            + (pynutil.delete(NEMO_SPACE) + delete_hour).ques
        )

        # east_german_dialects = (
        #     pynutil.add_weight(graph_um_full_hour, -0.01) | graph_quarter_hours
        # )
        east_german_dialects = graph_quarter_hours  # | graph_um_full_hour

        # Implements time zone logic
        # Assumes Universal Coordinated Time (UTC) as the default
        # Assumes whitespace separated characters in all abbreviations

        utc = pynini.accep("U T C") | pynini.accep("u t c")
        plus = pynini.cross("plus", "+")
        minus = pynini.cross("minus", "-")
        sign = plus | minus

        time_zone_values = (
            pynini.union(*[str(num) for num in range(0, 13)])
        ) @ graph_dozen.invert()

        UTC_timezones = (
            pynini.cross(utc, "UTC")
            + pynutil.delete(NEMO_SPACE)
            + sign
            + pynutil.delete(NEMO_SPACE)
            + time_zone_values.invert()
            + pynini.cross(" komma fünf", ".5").ques
        )

        remove_spaces = pynini.cdrewrite(
            pynutil.delete(NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA
        )
        capitalize = pynini.cdrewrite(TO_UPPER, "", "", NEMO_SIGMA)

        # The logic below is too general, resulting in capturing strings like "SMS", among others
        # A list of valid timezone abbreviations should be implemented instead
        # other_timezones = (
        #     pynini.closure((NEMO_ALPHA + pynini.accep(NEMO_SPACE)), 2, 2) + NEMO_ALPHA
        # )
        # other_timezones @= remove_spaces @ capitalize

        graph_timezone = (
            pynutil.insert("zone:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + (UTC_timezones)  # | other_timezones)
            + pynutil.insert('"')
        )

        graph_time_colloquial = (
            graph_half_hour
            | graph_min_or_sec_past_hour
            | graph_min_or_sec_to_hour
            | east_german_dialects
        )

        graph_time = (
            (
                graph_time_formal
                | graph_time_colloquial
                | pynutil.add_weight((graph_time_formal_no_unit), 2.0)
            )
            + (pynini.accep(NEMO_SPACE) + graph_postpositions).ques
            + (pynini.accep(NEMO_SPACE) + graph_timezone).ques
        )

        # Implements prepositional time expressions
        # The logic below will also capture time ranges (i.e. repeated PPs) and chained time expressions

        temp_prep_and_conj = pynini.string_file(get_abs_path("data/time/prep_conj.tsv"))

        graph_PPs = (
            # Prepositions aren't morphosyntax.
            # morphosyntactic_features is the only string field left in Sparrowhawk's TIME class
            pynutil.insert("morphosyntactic_features:")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert('"')
            + temp_prep_and_conj
            + pynutil.insert('"')
            + pynini.accep(NEMO_SPACE)
            + graph_time
        )

        # graph_time_final = graph_time | pynutil.add_weight(graph_PPs, 100.0)
        graph_time_final = graph_time | graph_PPs
        graph = self.add_tokens(graph_time_final)
        self.fst = graph.optimize()
