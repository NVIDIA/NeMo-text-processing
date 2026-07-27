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
from nemo_text_processing.text_normalization.ar.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_preserve_order,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time in a regular (formal) style, e.g.
        'time { hours: "7" minutes: "12" }' -> السابعة واثنتي عشرة دقيقة
        'time { hours: "1" minutes: "1" }' -> الواحدة ودقيقة واحدة
        'time { hours: "2" minutes: "2" }' -> الثانية ودقيقتان
        'time { hours: "3" minutes: "30" }' -> الثالثة وثلاثين دقيقة
        'time { hours: "18" minutes: "30" }' -> الثامنة عشرة وثلاثين دقيقة
        'time { hours: "5" minutes: "10" seconds: "2" preserve_order: true }' -> الخامسة وعشر دقائق وثانيتان
        'time { hours: "9" suffix: "صباحًا" }' -> التاسعة صباحًا

    The hour is a feminine ordinal (24-hour), the minutes/seconds are regular
    cardinals with number agreement (دقيقة/دقيقتان/دقائق). No colloquial
    ربع/نصف/إلا forms are produced, so the mapping is deterministic and consistent
    across h:m and h:m:s.

    Args:
        cardinal_tagger: cardinal tagger GraphFst (provides the number verbalization graph)
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        n3_10 = pynini.union(*[str(x) for x in range(2, 11)])
        n13_19 = pynini.union(*[str(x) for x in range(11, 20)])

        ordinals = pynini.string_file(get_abs_path("data/ordinal/ordinals.tsv"))
        mas_3_10 = pynini.string_file(get_abs_path("data/number/3_10mas.tsv"))
        graph_13_19 = pynini.string_file(get_abs_path("data/number/13_19.tsv"))
        time_zone_graph = pynini.invert(
            convert_space(
                pynini.union(*[x[1] for x in load_labels(get_abs_path("data/time/time_zone.tsv"))])
            )
        )
        number_verbalization = cardinal_tagger.graph

        # hour -> feminine ordinal; midnight hour "0" reads as الثانية عشرة
        hour = pynutil.delete('hours: "') + (pynini.cross("0", "12") | pynini.closure(NEMO_DIGIT, 1)) + pynutil.delete('"')
        hour_verbalized = pynutil.add_weight(hour @ ordinals, weight=0.001)

        # minutes: 3-10 -> plural دقائق, 12-19 -> teens + دقيقة, else -> cardinal + دقيقة
        minute_singular = (
            pynutil.delete('minutes: "')
            + pynini.closure(NEMO_DIGIT, 1)
            @ number_verbalization
            @ pynini.cdrewrite(pynini.cross("اثنا عشر", "اثنتا عشرة"), "", "", NEMO_SIGMA)
            @ pynini.cdrewrite(pynini.cross("احد عشر", "احدى عشرة"), "", "", NEMO_SIGMA)
            + pynutil.insert(" دقيقة")
            + pynutil.delete('"')
        )
        minute_plural = (
            pynutil.delete('minutes: "')
            + n3_10 @ pynini.closure(NEMO_DIGIT, 1) @ mas_3_10
            + pynutil.insert(" دقائق")
            + pynutil.delete('"')
        )
        minute_13_19 = (
            pynutil.delete('minutes: "')
            + n13_19 @ pynini.closure(NEMO_DIGIT, 1) @ graph_13_19
            + pynutil.insert(" دقيقة")
            + pynutil.delete('"')
        )
        # Reverse (polarity) agreement for the ones digit of a 21-59 compound before a
        # feminine counted noun (دقيقة/ثانية): the cardinal graph emits the masculine
        # citation form (خمسة وأربعين), but a feminine noun requires the feminine form
        # (خمس وأربعين). Fires only when the ones word is immediately followed by " و"
        # (the tens connector), so standalone/teens/plural forms are left untouched.
        feminize_unit = pynini.cdrewrite(
            pynini.cross("واحد", "إحدى")
            | pynini.cross("اثنين", "اثنتان")
            | pynini.cross("ثلاثة", "ثلاث")
            | pynini.cross("أربعة", "أربع")
            | pynini.cross("خمسة", "خمس")
            | pynini.cross("ستة", "ست")
            | pynini.cross("سبعة", "سبع")
            | pynini.cross("ثمانية", "ثمان")
            | pynini.cross("تسعة", "تسع"),
            "",
            " و",
            NEMO_SIGMA,
        )

        # The minutes/seconds number is the conjoined predicate of the time phrase and is
        # therefore nominative (عشرون/ثلاثون/...), while the cardinal graph defaults to the
        # genitive/accusative form (عشرين/ثلاثين/...). Applied to minutes/seconds only, not
        # the hour ordinal (e.g. الثالثة والعشرين for the 23rd hour stays unchanged).
        nominative_tens = pynini.cdrewrite(
            pynini.cross("عشرين", "عشرون")
            | pynini.cross("ثلاثين", "ثلاثون")
            | pynini.cross("أربعين", "أربعون")
            | pynini.cross("خمسين", "خمسون"),
            "",
            "",
            NEMO_SIGMA,
        )

        minute_verbalized = (
            (minute_plural | minute_13_19 | pynutil.add_weight(minute_singular, 0.001)) @ feminize_unit @ nominative_tens
        )

        # constrain suffix to the known suffix.tsv values (not an open NEMO_NOT_QUOTE
        # catch-all) so that inverting this verbalizer for ITN does not greedily absorb
        # the minutes/seconds text as a "suffix".
        suffix_values = pynini.union(*[x[1] for x in load_labels(get_abs_path("data/time/suffix.tsv"))])
        zone = pynutil.delete('zone: "') + time_zone_graph + pynutil.delete('"')
        suffix = pynutil.delete('suffix: "') + suffix_values + pynutil.delete('"')
        optional_suffix = pynini.closure(pynini.accep(" ") + suffix, 0, 1)
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)

        second_singular = (
            pynutil.delete('seconds: "')
            + pynini.closure(NEMO_DIGIT, 1)
            @ number_verbalization
            @ pynini.cdrewrite(pynini.cross("اثنا عشر", "اثنتا عشرة"), "", "", NEMO_SIGMA)
            @ pynini.cdrewrite(pynini.cross("احد عشر", "احدى عشرة"), "", "", NEMO_SIGMA)
            + pynutil.insert(" ثانية")
            + pynutil.delete('"')
        )
        second_plural = (
            pynutil.delete('seconds: "')
            + (n3_10 @ pynini.closure(NEMO_DIGIT, 1) @ mas_3_10 + pynutil.insert(" ثواني"))
            + pynutil.delete('"')
        )
        second_13_19 = (
            pynutil.delete('seconds: "')
            + n13_19 @ pynini.closure(NEMO_DIGIT, 1) @ graph_13_19
            + pynutil.insert(" ثانية")
            + pynutil.delete('"')
        )
        second_verbalized = (
            (second_plural | second_13_19 | pynutil.add_weight(second_singular, 0.001)) @ feminize_unit @ nominative_tens
        )

        # agreement fixes for 1/2 minute and second
        agreement = pynini.cdrewrite(
            pynini.cross("واحد دقيقة", "دقيقة واحدة")
            | pynini.cross("واحد ثانية", "ثانية واحدة")
            | pynini.cross("اثنين دقيقة", "دقيقتان")
            | pynini.cross("اثنين ثانية", "ثانيتان"),
            "",
            "",
            NEMO_SIGMA,
        )

        graph_hms = (
            hour_verbalized
            + pynini.accep(" ")
            + pynutil.insert("و")
            + minute_verbalized
            + pynini.accep(" ")
            + pynutil.insert("و")
            + second_verbalized
        )
        graph_hms @= agreement

        graph_hm = hour_verbalized + pynini.accep(" ") + pynutil.insert("و") + minute_verbalized
        graph_hm @= agreement

        # whole hour only (minutes were dropped by the tagger)
        graph_h = hour_verbalized

        self.graph = (graph_hms | graph_h | graph_hm) + optional_suffix + optional_zone
        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
