# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.se.graph_utils import TO_LOWER
from nemo_text_processing.text_normalization.se.utils import get_abs_path, load_labels

delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "skábmamánu 8. b. 1956" -> date { month: "skábmamánnu" day: "gávccát" year: "duhátovccičuođivihttalogiguhtta" preserve_order: true }
        "8/11/1956" -> date { day: "gávccát" month: "skábmamánnu" year: "duhátovccičuođivihttalogiguhtta" }
        "8-11 1956" -> date { day: "gávccát" month: "skábmamánnu" year: "duhátovccičuođivihttalogiguhtta" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        month_abbr_graph = load_labels(get_abs_path("data/dates/months_abbr.tsv"))
        number_to_month = pynini.string_file(get_abs_path("data/dates/numbers.tsv")).optimize()
        if not deterministic:
            number_to_month |= pynini.cross("1", "ođđajagemánnu")

        month_names = pynini.project(number_to_month, "output")
        month_nom_to_gen_map = pynini.string_map([("mánnu", "mánu")])
        self.months_nom2gen = month_names @ pynini.cdrewrite(month_nom_to_gen_map, "", "[EOS]", NEMO_SIGMA)
        self.months_gen2nom = pynini.invert(self.months_nom2gen)
        self.months_num2gen = (number_to_month @ self.months_nom2gen).optimize()
        month_graph = self.months_gen2nom
        month_graph |= pynini.string_file(get_abs_path("data/dates/months_gen.tsv"))

        month_abbr_graph = pynini.string_map(month_abbr_graph)
        month_abbr_graph = (
            pynutil.add_weight(month_abbr_graph, weight=0.0001)
            | ((TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete(".", weight=-0.0001), 0, 1)

        self.month_abbr = month_abbr_graph
        month_graph |= (TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_graph
        month_graph |= month_abbr_graph

        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ ordinal.graph_bare_ordinals
        day = (pynutil.insert("day: \"") + digit_day + pynutil.insert("\"")).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        graph_number_to_month = digit_month @ number_to_month

        month_name = (pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")).optimize()
        month_number = (pynutil.insert("month: \"") + graph_number_to_month + pynutil.insert("\"")).optimize()

        year = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 1, 3)  # 90, 990, 1990
        year @= cardinal.year | pynutil.add_weight(cardinal.graph, 0.1)
        self.year = year.optimize()

        year_only = pynutil.insert("year: \"") + year + pynutil.insert("\"")

        beaivi = pynutil.delete(pynini.union("b.", "beaivi"))
        preserve_order = pynutil.insert(" preserve_order: true")

        graph_md = month_name + NEMO_SPACE + day + pynini.closure(pynutil.delete("."), 0, 1) + NEMO_SPACE + beaivi
        self.md = (graph_md + preserve_order).optimize()

        graph_mdy = graph_md + pynini.closure(pynini.accep(" ") + year_only, 0, 1) + preserve_order
        self.mdy = graph_mdy.optimize()

        graph_dmy = pynini.Fst()
        for separator in ["/", "."]:
            graph_dmy |= (
                day
                + pynutil.delete(separator)
                + insert_space
                + month_number
                + pynini.closure(pynutil.delete(separator) + insert_space + year_only, 0, 1)
            )
        self.dmy = graph_dmy.optimize()
        graph_ymd = (
            year_only
            + pynutil.delete("/")
            + insert_space
            + month_number
            + pynini.closure(pynutil.delete("/") + insert_space + day, 0, 1)
        )
        self.ymd = graph_ymd.optimize()
        graph_ymd |= (
            year_only + pynutil.delete("-") + insert_space + month_number + pynutil.delete("-") + insert_space + day
        )

        separators = ["/", "-"]
        for sep in separators:
            year_optional = pynini.closure(NEMO_SPACE + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, " ") + month_number + year_optional
            graph_dmy |= new_graph

        final_graph = graph_dmy
        # final_graph |= year_only
        final_graph |= graph_ymd
        final_graph |= graph_mdy

        self.final_graph = final_graph.optimize()

        month_genitive = pynini.project(self.months_nom2gen, "output")
        month_genitive |= pynini.project(pynini.string_file(get_abs_path("data/dates/months_gen.tsv")), "input")
        interval_first = (
            optional_leading_zero @ pynini.union(*[str(x) for x in range(10, 32)]) @ ordinal.graphs["loc_sg"]
        )
        interval_last = (
            optional_leading_zero @ pynini.union(*[str(x) for x in range(10, 32)]) @ ordinal.graphs["ill_sg"]
        )
        day_interval = (
            month_genitive
            + NEMO_SPACE
            + interval_first
            + pynini.cross(".-", " beaivvis ")
            + interval_last
            + pynutil.delete(".")
            + pynutil.delete(" ")
            + pynutil.delete("beaivvit")
        )

        abbreviated_month_genitive = month_abbr_graph @ self.months_nom2gen
        month_interval = (
            abbreviated_month_genitive
            + NEMO_SPACE
            + interval_first
            + pynutil.delete(". b. - ")
            + pynutil.insert(" beaivvis ")
            + abbreviated_month_genitive
            + NEMO_SPACE
            + interval_last
            + pynutil.delete(". b.")
            + pynutil.insert(" beaivái")
        )

        two_digit_year = (NEMO_DIGIT**2) @ cardinal.graph_with_leading_zero
        year_range = (
            pynini.accep("jagit")
            + NEMO_SPACE
            + cardinal.year
            + pynini.cross("-", " gitta ")
            + (two_digit_year | cardinal.year | cardinal.graph_with_leading_zero)
        )
        name_graph = (
            pynutil.insert('name: "') + convert_space(day_interval | month_interval | year_range) + pynutil.insert('"')
        )
        self.fst = (self.add_tokens(self.final_graph) | name_graph).optimize()
