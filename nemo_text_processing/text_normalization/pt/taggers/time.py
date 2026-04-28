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

from nemo_text_processing.text_normalization.pt.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese (Brazilian) time, e.g.
        14:30 -> time { hours: "catorze" minutes: "trinta" preserve_order: true }
        14:30:05 -> time { hours: "catorze" minutes: "trinta" seconds: "cinco" preserve_order: true }
        09:00:31 -> time { hours: "nove" minutes: "zero" seconds: "trinta e um" preserve_order: true }
        12:00 -> time { hours: "doze" preserve_order: true }
        11:00 da manhã -> time { hours: "onze" suffix: "da manhã" preserve_order: true }
        16:00 da tarde -> time { hours: "quatro" suffix: "da tarde" preserve_order: true }
        23:18 da tarde -> time { hours: "vinte e três" ... suffix: "da tarde" preserve_order: true }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """


    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph.optimize()

        hour_words = []
        for h in range(24):
            key = str(h)
            comp = pynini.compose(pynini.accep(key), cardinal_graph).optimize()
            hour_words.append(pynini.shortestpath(comp, nshortest=1, unique=True).string())

        hour_delete_fsts = []
        for h in range(24):
            if h < 10:
                hour_delete_fsts.append(
                    pynini.union(pynutil.delete(str(h)), pynutil.delete(f"0{h}")).optimize()
                )
            else:
                hour_delete_fsts.append(pynutil.delete(str(h)))

        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal_graph
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal_graph
        final_graph_minute = (
            pynutil.insert('minutes: "')
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert('"')
        )

        final_graph_second = (
            pynutil.insert('seconds: "')
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert('"')
        )

        # HMS verbalizer always expects ``minutes`` and ``seconds`` tags; bare ``delete("00")`` omits them.
        zero_word = hour_words[0]
        minutes_zero = (
            pynutil.delete("00")
            + pynutil.insert('minutes: "')
            + pynutil.insert(zero_word)
            + pynutil.insert('"')
        )
        seconds_zero = (
            pynutil.delete("00")
            + pynutil.insert('seconds: "')
            + pynutil.insert(zero_word)
            + pynutil.insert('"')
        )

        delete_h = pynini.union(
            pynutil.delete(pynini.accep(pynini.escape("h"))),
            pynutil.delete(pynini.accep(pynini.escape("H"))),
        )

        time_delim = pynini.union(
            pynini.accep(pynini.escape(":")),
            pynini.accep(pynini.escape(".")),
        )

        period_rows = load_labels(get_abs_path("data/time/day_period_suffix.tsv"))
        period_meta = []
        for row in period_rows:
            if len(row) < 2 or not row[0].strip():
                continue
            tail, tag_val = row[0].strip(), row[1].strip()
            if len(row) < 4 or not row[2].strip().isdigit() or not row[3].strip().isdigit():
                raise ValueError(
                    f"day_period_suffix.tsv row must have 4 columns (tail, tag, hour_min, hour_max): {row!r}"
                )
            h0, h1 = int(row[2].strip()), int(row[3].strip())
            allowed = frozenset(range(h0, h1 + 1))
            suf_fst = insert_space + delete_space + pynutil.delete("da") + delete_space + pynutil.delete(tail)
            period_meta.append((tag_val, allowed, suf_fst, tail))

        preserve = pynutil.insert(" preserve_order: true")

        mid_hm = pynutil.delete(time_delim) + (pynutil.delete("00") | insert_space + final_graph_minute)
        mid_h_minute = delete_h + (pynutil.delete("00") | insert_space + final_graph_minute)
        mid_h_only = delete_h
        mid_hms = (
            pynutil.delete(time_delim)
            + (minutes_zero | insert_space + final_graph_minute)
            + pynutil.delete(time_delim)
            + (seconds_zero | insert_space + final_graph_second)
        )

        graph_chunks = []
        for mid_after_hour in (mid_hm, mid_h_minute, mid_h_only, mid_hms):
            branches = []
            for h in range(24):
                hd = hour_delete_fsts[h]
                hw24 = hour_words[h]
                hour_tok_24 = pynutil.insert('hours: "') + pynutil.insert(hw24) + pynutil.insert('"')
                branches.append(hd + hour_tok_24 + mid_after_hour + preserve)
                for tag_val, allowed, suf, tail in period_meta:
                    keep_suffix, hour_idx = TimeFst._resolve_suffix_hour(h, tail, allowed)
                    hw_suf = hour_words[hour_idx]
                    hour_tok_suf = pynutil.insert('hours: "') + pynutil.insert(hw_suf) + pynutil.insert('"')
                    if keep_suffix:
                        branches.append(
                            hd
                            + hour_tok_suf
                            + mid_after_hour
                            + suf
                            + pynutil.insert(f' suffix: "{tag_val}"')
                            + preserve
                        )
                    else:
                        # User wrote a period: always emit ``suffix:`` so TN does not drop it from speech
                        # (hours stay 24h when the period does not match the clock policy).
                        branches.append(
                            hd
                            + hour_tok_24
                            + mid_after_hour
                            + suf
                            + pynutil.insert(f' suffix: "{tag_val}"')
                            + preserve
                        )
            graph_chunks.append(pynini.union(*branches).optimize())

        final_graph = pynini.union(*graph_chunks).optimize()
        self.fst = self.add_tokens(final_graph).optimize()


    @staticmethod
    def _resolve_suffix_hour(h: int, period_tail: str, allowed: frozenset) -> tuple[bool, int]:
        """Return (keep_suffix, hour_index) for ``hour_words[hour_index]`` when a day-period applies."""
        if period_tail == "manhã":
            allowed_m = allowed | frozenset({1, 2, 3, 4, 5})
            if h not in allowed_m:
                return False, h
            return True, h
        if period_tail == "tarde":
            if h in allowed:
                return True, 12 if h == 12 else h - 12
            if 1 <= h <= 5 and (h + 12) in allowed:
                return True, h
            return False, h
        if period_tail == "noite":
            if h in allowed:
                return True, h - 12
            if 6 <= h <= 11 and (h + 12) in allowed:
                return True, h
            return False, h
        if period_tail == "madrugada":
            if h in allowed:
                return True, h
            return False, h
        return False, h