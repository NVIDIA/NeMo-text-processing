# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ta.graph_utils import (
    AM_WORD,
    DAY_PART_ABBREVIATIONS,
    DAY_PARTS,
    NEMO_TA_DIGIT,
    NEMO_TA_NON_ZERO,
    NEMO_TA_ZERO,
    PM_WORD,
    TO_TA_DIGITS,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

# The verbalizer speaks மணி itself, so a written மணி/மணிக்கு and a bare case suffix on the
# digits (3:30க்கு, 10:30 இல்) are absorbed rather than carried as a field.
HOUR_NOUNS = ("மணிக்கு", "மணி")
ABSORBED_SUFFIXES = ("க்கு", "ல்", "இல்")
# Hour 24 is only meaningful as 24:00.
EXACT_ONLY_HOURS = ("இருபத்துநான்கு",)


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        12:30:30 -> time { hours: "பன்னிரண்டு" minutes: "முப்பது" seconds: "முப்பது" }
        1:40 -> time { hours: "ஒரு" minutes: "நாற்பது" }
        10:00க்கு -> time { hours: "பத்து" }
        10:30 AM -> time { hours: "பத்து" minutes: "முப்பது" meridiem: "முற்பகல்" }
        காலை 10.30 -> time { hours: "பத்து" minutes: "முப்பது" meridiem: "காலை" }

    Reads ``data/time/hours.tsv``, ``data/time/minutes.tsv`` and ``data/time/seconds.tsv`` (Tamil
    digits to words; hours 0-24, minutes and seconds 01-59). A press-style dotted time (10.30)
    is only a time with a clock context: a trailing hour noun, or a day-part word before or
    after it.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
        seconds_graph = pynini.string_file(get_abs_path("data/time/seconds.tsv"))

        delete_colon = pynutil.delete(":")

        delete_leading_zero_native = (
            (NEMO_TA_NON_ZERO + NEMO_TA_DIGIT) | (pynutil.delete(NEMO_TA_ZERO) + NEMO_TA_DIGIT) | NEMO_TA_DIGIT
        ).optimize()
        delete_leading_zero_ascii = (
            (pynini.difference(NEMO_DIGIT, "0") + NEMO_DIGIT) | (pynutil.delete("0") + NEMO_DIGIT) | NEMO_DIGIT
        ).optimize()

        hour_input = (
            pynini.compose(delete_leading_zero_native, hours_graph)
            | pynini.compose(delete_leading_zero_ascii, TO_TA_DIGITS @ hours_graph)
        ).optimize()
        minute_input = (
            pynini.compose(pynini.closure(NEMO_TA_DIGIT, 1), minutes_graph)
            | pynini.compose(pynini.closure(NEMO_DIGIT, 1), TO_TA_DIGITS @ minutes_graph)
        ).optimize()
        second_input = (
            pynini.compose(pynini.closure(NEMO_TA_DIGIT, 1), seconds_graph)
            | pynini.compose(pynini.closure(NEMO_DIGIT, 1), TO_TA_DIGITS @ seconds_graph)
        ).optimize()

        hour_any = hour_input
        hour_input = hour_input @ pynini.difference(pynini.closure(NEMO_CHAR), pynini.union(*EXACT_ONLY_HOURS))
        self.hours = pynutil.insert("hours: \"") + hour_input + pynutil.insert("\" ")
        hours_any = pynutil.insert("hours: \"") + hour_any + pynutil.insert("\" ")
        self.minutes = pynutil.insert("minutes: \"") + minute_input + pynutil.insert("\" ")
        self.seconds = pynutil.insert("seconds: \"") + second_input + pynutil.insert("\" ")

        # A trailing written hour noun, or a case suffix the verbalizer does not attach, is
        # consumed silently, glued to the digits or spaced (10:30க்கு, 10:30 இல், 7:00 மணி).
        space = pynini.closure(NEMO_SPACE, 0, 1)
        hour_word_tail = space + pynutil.delete(pynini.union(*HOUR_NOUNS))
        absorbed_tail = hour_word_tail | space + pynutil.delete(pynini.union(*ABSORBED_SUFFIXES))
        optional_tail = pynini.closure(absorbed_tail, 0, 1).optimize()

        graph_hms = (
            self.hours
            + delete_colon
            + insert_space
            + self.minutes
            + delete_colon
            + insert_space
            + self.seconds
            + optional_tail
        )
        double_zero = pynini.union("00", NEMO_TA_ZERO + NEMO_TA_ZERO)
        delete_zero_seconds = pynini.closure(pynutil.delete(":" + double_zero), 0, 1)
        graph_hm = self.hours + delete_colon + insert_space + self.minutes + delete_zero_seconds + optional_tail
        delete_zero_minutes = delete_colon + pynutil.delete(double_zero)
        graph_h = hours_any + delete_zero_minutes + delete_zero_seconds + optional_tail
        # 10:00:30 keeps only the seconds.
        graph_h_s = self.hours + delete_zero_minutes + delete_colon + insert_space + self.seconds + optional_tail

        # Trailing AM/PM becomes a meridiem field the verbalizer fronts.
        meridiem_word = pynini.cross(pynini.union("AM", "am", "A.M.", "a.m."), AM_WORD) | pynini.cross(
            pynini.union("PM", "pm", "P.M.", "p.m."), PM_WORD
        )
        required_meridiem = (
            pynutil.delete(pynini.closure(" ", 0, 1))
            + pynutil.insert("meridiem: \"")
            + meridiem_word
            + pynutil.insert("\" ")
        )
        meridiem = pynini.closure(required_meridiem, 0, 1)

        final_graph = (
            graph_hms
            | pynutil.add_weight(graph_hm, 1.0)
            | pynutil.add_weight(graph_h_s, 1.0)
            | pynutil.add_weight(graph_h, 0.8)
        ) + meridiem

        # A bare hour with AM/PM is a clock time: 7 AM, 7pm.
        final_graph |= pynutil.add_weight(self.hours + required_meridiem, 0.9)

        # Press-style dotted time (10.30) is only a time with a clock context.
        two_digit_minutes = pynini.compose(
            pynini.union(NEMO_TA_DIGIT + NEMO_TA_DIGIT, NEMO_DIGIT + NEMO_DIGIT), minute_input
        )
        dotted = (
            self.hours
            + pynutil.delete(".")
            + insert_space
            + pynutil.insert("minutes: \"")
            + two_digit_minutes
            + pynutil.insert("\" ")
        )
        # 6.00 reads as the bare hour.
        dotted |= self.hours + pynutil.delete("." + double_zero)
        dotted_tail = pynini.closure(hour_word_tail, 0, 1)
        contexts = [(w, w) for w in DAY_PARTS] + list(DAY_PART_ABBREVIATIONS.items())
        dotted_graphs = [dotted + hour_word_tail, dotted + dotted_tail + required_meridiem]
        for written, spoken in contexts:
            meridiem_field = pynutil.insert(f"meridiem: \"{spoken}\" ")
            dotted_graphs.append(pynutil.delete(written) + pynutil.delete(" ") + dotted + dotted_tail + meridiem_field)
            dotted_graphs.append(dotted + space + pynutil.delete(written) + meridiem_field)
        # A cued dotted time must outrank a measure reading of the same span (10.30 மணி).
        final_graph |= pynutil.add_weight(pynini.union(*dotted_graphs), -2.5)

        self.fst = self.add_tokens(final_graph).optimize()
