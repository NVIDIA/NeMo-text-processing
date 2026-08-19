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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class DateFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese (Brazilian) dates, e.g.
        15/03/2024 -> date { day: "quinze" month: "março" year: "dois mil e vinte e quatro" preserve_order: true }
        15 de março de 2024 -> date { day: "quinze" month: "março" year: "dois mil e vinte e quatro" preserve_order: true }
        2024-03-15 -> date { day: "quinze" month: "março" year: "dois mil e vinte e quatro" preserve_order: true }
        03/15/2024 -> date { day: "quinze" month: "março" year: "dois mil e vinte e quatro" preserve_order: true }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)
        numbers = cardinal.graph

        month_rows = load_labels(get_abs_path("data/date/months.tsv"))
        month_pairs = [(r[0], r[1]) for r in month_rows if len(r) >= 2]
        month_to_word = pynini.string_map(month_pairs).optimize()

        day_10_31 = ((NEMO_DIGIT - "0") + NEMO_DIGIT) @ pynini.union(*[str(x) for x in range(10, 32)]) @ numbers
        day_02_09 = pynutil.delete("0") + (pynini.union(*[str(x) for x in range(2, 10)]) @ numbers)
        day_2_9 = pynini.union(*[str(x) for x in range(2, 10)]) @ numbers
        day_inner = pynini.union(
            pynini.cross("01", "primeiro"),
            day_10_31,
            day_02_09,
            day_2_9,
            pynini.cross("1", "primeiro"),
        ).optimize()
        day_part = pynutil.insert('day: "') + day_inner + pynutil.insert('"')

        month_digits = (
            pynini.union("10", "11", "12")
            | pynutil.delete("0") + pynini.union(*[str(x) for x in range(1, 10)])
            | pynini.union(*[str(x) for x in range(1, 10)])
        )
        month_num = month_digits @ month_to_word
        month_part = pynutil.insert('month: "') + month_num + pynutil.insert('"')

        year_num = ((NEMO_DIGIT - "0") + NEMO_DIGIT**3) @ numbers
        year_part = pynutil.insert('year: "') + year_num + pynutil.insert('"')

        preserve = pynutil.insert(" preserve_order: true")

        delete_de = delete_space + pynutil.delete("de") + delete_space
        month_names = sorted({r[1] for r in month_rows if len(r) >= 2}, key=len, reverse=True)
        text_pairs = []
        for name in month_names:
            text_pairs.append((name, name))
            if name and name[0].islower():
                text_pairs.append((name[0].upper() + name[1:], name))
        month_written = pynutil.insert('month: "') + pynini.string_map(text_pairs).optimize() + pynutil.insert('"')
        graph_text = day_part + delete_de + month_written + delete_de + year_part + preserve

        sep_path = get_abs_path("data/date/numeric_separators.tsv")
        separators = [r[0].strip() for r in load_labels(sep_path) if r and r[0].strip()]

        one_or_two_digits = pynini.closure(NEMO_DIGIT, 1, 2)
        year_four = (NEMO_DIGIT - "0") + NEMO_DIGIT**3
        _mdy_weight = 0.05

        months_spoken = sorted({r[1] for r in month_rows if len(r) >= 2})
        day_spokens = set()
        for n in range(1, 32):
            for key in (str(n), f"{n:02d}"):
                dstr = pynini.shortestpath(pynini.compose(pynini.accep(key), day_inner.optimize())).string()
                day_spokens.add(dstr)

        _preserve_tail = " preserve_order: true"

        ymd_to_dmy_graph = None
        mdy_to_dmy_graph = None
        for month in months_spoken:
            for day in day_spokens:
                # After year: + sigma (year value + quotes), delete month/day and trailing preserve
                # so the input is fully consumed (mdy_to_dmy does not need this: sigma eats the tail).
                ymd_curr = (
                    pynutil.insert('day: "' + day + '" month: "' + month + '" ')
                    + pynini.accep("year:")
                    + NEMO_SIGMA
                    + pynutil.delete(' month: "' + month + '" day: "' + day + '"' + _preserve_tail)
                )
                ymd_to_dmy_graph = ymd_curr if ymd_to_dmy_graph is None else pynini.union(ymd_to_dmy_graph, ymd_curr)

                mdy_curr = (
                    pynutil.insert('day: "' + day + '" month: "' + month + '" ')
                    + pynutil.delete('month: "' + month + '" day: "' + day + '" ')
                    + pynini.accep("year:")
                    + NEMO_SIGMA
                )
                mdy_to_dmy_graph = mdy_curr if mdy_to_dmy_graph is None else pynini.union(mdy_to_dmy_graph, mdy_curr)

        ymd_to_dmy_graph = ymd_to_dmy_graph.optimize()
        mdy_to_dmy_graph = mdy_to_dmy_graph.optimize()

        patterns = [graph_text]
        for sep in separators:
            sep_accep = pynini.accep(pynini.escape(sep))
            del_sep = pynutil.delete(sep_accep)

            dmy_core = day_part + del_sep + insert_space + month_part + del_sep + insert_space + year_part + preserve
            iso_core = year_part + del_sep + insert_space + month_part + del_sep + insert_space + day_part + preserve
            mdy_core = month_part + del_sep + insert_space + day_part + del_sep + insert_space + year_part + preserve

            lhs_dmy = one_or_two_digits + sep_accep + one_or_two_digits + sep_accep + year_four
            lhs_iso = year_four + sep_accep + one_or_two_digits + sep_accep + one_or_two_digits
            lhs_mdy = one_or_two_digits + sep_accep + one_or_two_digits + sep_accep + year_four

            patterns.append(pynini.compose(lhs_dmy, dmy_core))
            patterns.append(
                pynutil.add_weight(
                    pynini.compose(
                        pynini.compose(lhs_mdy, mdy_core),
                        mdy_to_dmy_graph,
                    ),
                    _mdy_weight,
                )
            )
            patterns.append(
                pynini.compose(
                    pynini.compose(lhs_iso, iso_core),
                    ymd_to_dmy_graph,
                )
            )

        self.fst = self.add_tokens(pynini.union(*patterns).optimize()).optimize()
