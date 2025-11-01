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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_preserve_order,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic, e.g.
        time { hours: "2" minutes: "15"} -> "zwei uhr fünfzehn"
        time { minutes: "15" hours: "2" } -> "viertel nach zwei"
        time { minutes: "15" hours: "2" } -> "fünfzehn nach zwei"
        time { hours: "14" minutes: "15"} -> "vierzehn uhr fünfzehn"
        time { minutes: "15" hours: "14" } -> "viertel nach zwei"
        time { minutes: "15" hours: "14" } -> "fünfzehn nach drei"
        time { minutes: "45" hours: "14" } -> "viertel vor drei"

    Args:
        cardinal_tagger: cardinal_tagger tagger GraphFst
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        # add weight so when using inverse text normalization this conversion is depriotized
        night_to_early = pynutil.add_weight(
            pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to_night.tsv"))).optimize(), weight=0.0001
        )
        hour_to = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to.tsv"))).optimize()
        minute_to = pynini.invert(pynini.string_file(get_abs_path("data/time/minute_to.tsv"))).optimize()
        time_zone_graph = pynini.invert(
            convert_space(pynini.union(*[x[1] for x in load_labels(get_abs_path("data/time/time_zone.tsv"))]))
        )

        graph_zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv"))).optimize()
        number_verbalization = graph_zero | cardinal_tagger.two_digit_non_zero
        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        hour_verbalized = hour @ number_verbalization @ pynini.cdrewrite(
            pynini.cross("eins", "ein"), "[BOS]", "[EOS]", NEMO_SIGMA
        ) + pynutil.insert(" uhr")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        zone = pynutil.delete("zone: \"") + time_zone_graph + pynutil.delete("\"")
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)

        optional_has_uhr = pynini.closure(
            pynutil.delete('has_uhr: "')
            + pynini.union('true', 'True')
            + pynutil.delete('"')
            + pynini.closure(pynini.accep(" "), 0, 1),
            0,
            1,
        )

        graph_hms = (
            hour_verbalized
            + pynini.accep(" ")
            + minute @ number_verbalization
            + pynutil.insert(" minuten")
            + pynini.accep(" ")
            + second @ number_verbalization
            + pynutil.insert(" sekunden")
            + optional_zone
        )
        graph_hms @= pynini.cdrewrite(
            pynini.cross("eins minuten", "eine minute") | pynini.cross("eins sekunden", "eine sekunde"),
            pynini.union(" ", "[BOS]"),
            "",
            NEMO_SIGMA,
        )

        min_30 = [str(x) for x in range(1, 31)]
        min_30 = pynini.union(*min_30)
        min_29 = [str(x) for x in range(1, 30)]
        min_29 = pynini.union(*min_29)

        graph_h = hour_verbalized
        graph_hm = hour_verbalized + pynini.accep(" ") + minute @ number_verbalization

        graph_m_past_h = (
            minute @ min_30 @ (number_verbalization | pynini.cross("15", "viertel"))
            + pynini.accep(" ")
            + pynutil.insert("nach ")
            # + hour @ number_verbalization
            + hour @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", NEMO_SIGMA) @ number_verbalization
        )
        graph_m30_h = (
            minute @ pynini.cross("30", "halb")
            + pynini.accep(" ")
            + hour @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", NEMO_SIGMA) @ hour_to @ number_verbalization
        )
        graph_m_to_h = (
            minute @ minute_to @ min_29 @ (number_verbalization | pynini.cross("15", "viertel"))
            + pynini.accep(" ")
            + pynutil.insert("vor ")
            + hour @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", NEMO_SIGMA) @ hour_to @ number_verbalization
        )

        self.graph = (
            (
                graph_hms
                | graph_h
                | graph_hm
                | pynutil.add_weight(graph_m_past_h, weight=0.0001)
                | pynutil.add_weight(graph_m30_h, weight=0.0001)
                | pynutil.add_weight(graph_m_to_h, weight=0.0001)
            )
            + optional_zone
            + optional_has_uhr
        )

        # duration branch: restores the preposition surface at the front and only emits minutes/seconds when present.
        duration_cues = pynini.string_file(get_abs_path("data/time/duration_cues.tsv"))
        # accept NBSP and ASCII spaces after the closing quote
        space_after = pynini.union(" ", NEMO_NON_BREAKING_SPACE)
        preposition_field = (
            pynutil.delete('preposition: "')
            + (convert_space(duration_cues) | duration_cues)
            + pynutil.delete('"')
            + pynini.closure(space_after, 1, 1)
        )

        # numeric fields
        hour_num = pynutil.delete('hours: "') + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete('"')
        minute_num = pynutil.delete('minutes: "') + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete('"')
        second_num = pynutil.delete('seconds: "') + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete('"')

        hour_dur = hour_num @ number_verbalization + pynutil.insert(' stunden')
        minute_dur = minute_num @ number_verbalization + pynutil.insert(' minuten')
        second_dur = second_num @ number_verbalization + pynutil.insert(' sekunden')

        # optional minutes/seconds (only emitted if present in the token)
        optional_min = pynini.closure(pynini.accep(' ') + minute_dur, 0, 1)
        optional_sec = pynini.closure(pynini.accep(' ') + second_dur, 0, 1)

        duration_prep_hms = preposition_field + hour_dur + optional_min + optional_sec
        duration_prep_ms = preposition_field + minute_dur + optional_sec
        duration_graph = duration_prep_hms | duration_prep_ms

        self.graph = duration_graph | self.graph

        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
