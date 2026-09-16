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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import CardinalFst, half_form_rows
from nemo_text_processing.inverse_text_normalization.ta.taggers.decimal import FRACTION_MINUTES, quarter_form_graph
from nemo_text_processing.text_normalization.en.graph_utils import delete_space
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import DAY_PARTS, TO_ASCII_DIGITS
from nemo_text_processing.text_normalization.ta.utils import get_abs_path as tn_abs_path

CLOCK_MAX_HOUR = 23

# Bare "X மணி" is a duration (two hours); the hour-only form needs the dative மணிக்கு or a
# day-part word. அரை alone is half an hour, never half past twelve.
HOUR_NOUNS = ("மணிக்கு", "மணி")
MINUTE_NOUNS = ("நிமிடங்கள்", "நிமிடம்", "நிமிடத்திற்கு", "நிமிடத்தில்", "நிமிடத்துக்கு")
SECOND_NOUNS = ("வினாடிகள்", "வினாடி", "வினாடிக்கு", "வினாடியில்", "நொடி")
CLOCK_HOUR_NOUN = "மணிக்கு"
# The counting word for one minute or second (ஒரு நிமிடம் -> :01).
MINUTE_ONE = "ஒரு"


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying spoken times, e.g.
        பத்து மணி முப்பது நிமிடம் -> time { hours: "10" minutes: "30" preserve_order: true }
        பத்து மணிக்கு -> time { hours: "10" preserve_order: true }
        பத்தரை மணிக்கு -> time { hours: "10" minutes: "30" preserve_order: true }
        காலை பத்து மணி -> time { morphosyntactic_features: "காலை" hours: "10" preserve_order: true }

    A bare "X மணி" is a duration, so the hour-only form converts only with the dative மணிக்கு or a
    fronted day-part word, which travels as ``morphosyntactic_features``. Reads the TN
    ``data/time/{hours,minutes,seconds}.tsv`` tables from the spoken side.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="time", kind="classify")

        def table_words(name: str) -> 'pynini.FstLike':
            rows = [r for r in load_labels(tn_abs_path(f"data/time/{name}.tsv")) if len(r) >= 2]
            return (pynini.invert(pynini.string_map([(k, v) for k, v, *_ in rows])) @ TO_ASCII_DIGITS).optimize()

        # Any spoken number up to 23 may head a time; ஒரு (the clock one) counts as 1 here.
        hour_table = table_words("hours")
        one_word = pynini.project(cardinal.words_to_digits @ pynini.accep("1"), "input")
        hour_words = pynini.union(
            cardinal.words_to_digits_licensed, cardinal.read(hour_table | pynini.cross(one_word, "1"))
        ).optimize()
        clock_hours = pynini.union(*[str(h) for h in range(CLOCK_MAX_HOUR + 1)])
        hour_words = (hour_words @ clock_hours).optimize()
        minute_words = cardinal.read(table_words("minutes") | pynini.cross(MINUTE_ONE, "01"))
        second_words = cardinal.read(table_words("seconds") | pynini.cross(MINUTE_ONE, "01"))

        hour_plain = pynutil.delete(pynini.union(*HOUR_NOUNS))
        minute_plain = pynutil.delete(pynini.union(*MINUTE_NOUNS))
        second_plain = pynutil.delete(pynini.union(*SECOND_NOUNS))
        clock_noun = delete_space + pynutil.delete(CLOCK_HOUR_NOUN)

        hours = pynutil.insert("hours: \"") + hour_words + pynutil.insert("\"")
        minutes = pynutil.insert(" minutes: \"") + minute_words + pynutil.insert("\"")
        seconds = pynutil.insert(" seconds: \"") + second_words + pynutil.insert("\"")

        graph_h = hours + clock_noun
        graph_hm = hours + delete_space + hour_plain + delete_space + minutes + delete_space + minute_plain
        graph_hms = graph_hm + delete_space + seconds + delete_space + second_plain
        graph_hs = hours + delete_space + hour_plain + delete_space + seconds + delete_space + second_plain
        # Hour and minute with no hour noun between them, as ASR often renders a clock time:
        # பத்து முப்பது மணிக்கு -> 10:30. The dative is required, so a bare pair stays a number.
        graph_hm_bare = hours + delete_space + minutes + clock_noun

        graph = graph_hms | graph_hm | graph_hs | graph_h | pynutil.add_weight(graph_hm_bare, 0.1)

        # Half- and quarter-hour idioms: பத்தரை மணிக்கு -> 10:30, பத்தே கால் மணிக்கு -> 10:15. Bare
        # "Xரை மணி" is a duration (2.5 hours), so the clock reading needs the dative.
        fused = pynini.union(
            *[
                pynini.cross(word, f"hours: \"{ip}\" minutes: \"{FRACTION_MINUTES[fp]}\"")
                for word, ip, fp, *_ in half_form_rows()
                if fp in FRACTION_MINUTES and ip != "0" and int(ip) <= CLOCK_MAX_HOUR
            ]
        )
        fused |= quarter_form_graph(
            hour_words, "hours: \"", "\"", lambda fraction: f" minutes: \"{FRACTION_MINUTES[fraction]}\""
        )
        graph |= fused + clock_noun

        # A fronted day-part word travels with the time and makes even a bare "X மணி" a clock
        # time (காலை பத்து மணி -> காலை 10:00).
        day_part = (
            pynutil.insert("morphosyntactic_features: \"")
            + pynini.union(*DAY_PARTS)
            + pynutil.insert("\" ")
            + pynutil.delete(" ")
        )
        bare_hour = (hours | fused) + delete_space + hour_plain
        graph = pynini.closure(day_part, 0, 1) + graph | day_part + bare_hour
        graph += pynutil.insert(" preserve_order: true")
        self.fst = self.add_tokens(graph).optimize()
