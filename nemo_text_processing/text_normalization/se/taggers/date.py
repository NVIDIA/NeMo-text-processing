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
from nemo_text_processing.text_normalization.se.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.se.graph_utils import TO_LOWER
from pynini.lib import pynutil

graph_teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv"))).optimize()
graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
ties_graph = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv"))).optimize()
delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT


def get_year_graph(cardinal: GraphFst) -> 'pynini.FstLike':
    """
    Returns year verbalizations as fst

     < 2000 neunzehn (hundert) (vier und zwanzig), >= 2000 regular cardinal
    **00 ** hundert

    Args:
        delete_leading_zero: removed leading zero
        cardinal: cardinal GraphFst
    """

    year_gt_2000 = (pynini.union("21", "20") + NEMO_DIGIT ** 2) @ cardinal.graph

    graph_two_digit = delete_leading_zero @ cardinal.two_digit_non_zero
    hundred = pynutil.insert("hundert")
    graph_double_double = (
        (pynini.accep("1") + NEMO_DIGIT) @ graph_two_digit
        + insert_space
        + pynini.closure(hundred + insert_space, 0, 1)
        + graph_two_digit
    )
    # for 20**
    graph_double_double |= pynini.accep("20") @ graph_two_digit + insert_space + graph_two_digit
    graph = (
        graph_double_double
        | (pynini.accep("1") + NEMO_DIGIT) @ graph_two_digit + insert_space + pynutil.delete("00") + hundred
        | year_gt_2000
    )
    return graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g. 
        "skábmamánu 8. b. 1956" -> date { month: "skábmamánnu" day: "gávccát" year: "1956" preserve_order: true }
        "1994" -> date { year: "neunzehn vier und neuzig" }
        "1900" -> date { year: "neunzehn hundert" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        month_abbr_graph = load_labels(get_abs_path("data/months/months_abbr.tsv"))
        number_to_month = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        if not deterministic:
            number_to_month |= pynini.cross("1", "ođđajagemánnu")

        month_graph = pynini.project(number_to_month, "output")
        month_nom_to_gen_map = pynini.string_map([("mánnu", "mánu")])
        self.months_nom2gen = month_graph @ pynini.cdrewrite(month_nom_to_gen_map, "", "[EOS]", NEMO_SIGMA)
        self.months_gen2nom = pynini.invert(self.months_nom2gen)

        month_abbr_graph = pynini.string_map(month_abbr_graph)
        month_abbr_graph = (
            pynutil.add_weight(month_abbr_graph, weight=0.0001)
            | ((TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete(".", weight=-0.0001), 0, 1)

        self.month_abbr = month_abbr_graph
        month_graph |= (TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_graph
        month_graph |= month_abbr_graph

        numbers = cardinal.graph_hundred_component_at_least_one_none_zero_digit
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ ordinal.graph_bare_ordinals
        day = (pynutil.insert("day: \"") + digit_day + pynutil.insert("\"")).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month
        digit_month @= numbers

        month_name = (pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")).optimize()
        month_number = (
            pynutil.insert("month: \"")
            + (pynutil.add_weight(digit_month, weight=0.0001) | number_to_month)
            + pynutil.insert("\"")
        ).optimize()

        # prefer cardinal over year
        year = pynutil.add_weight(get_year_graph(cardinal=cardinal), weight=0.001)
        self.year = year

        year_only = pynutil.insert("year: \"") + year + pynutil.insert("\"")

        graph_dmy = (
            day
            + pynutil.delete(".")
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + insert_space
            + month_name
            + pynini.closure(pynini.accep(" ") + year_only, 0, 1)
        )

        beaivi = pynini.union("b.", "beaivi")
        preserve_order = pynutil.insert(" preserve_order: true")

        graph_md = (
            month_name
            + pynini.closure(pynutil.delete("."), 0, 1)
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + day
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + beaivi
        )
        self.md = (graph_md + preserve_order).optimize()

        graph_mdy = (
            graph_md
            + pynini.closure(pynini.accep(" ") + year_only, 0, 1)
            + preserve_order
        )
        self.mdy = graph_mdy

        separators = ["."]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, " ") + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, " ") + month_number + year_optional
            graph_dmy |= new_graph

        dash = "-"
        day_optional = pynini.closure(pynini.cross(dash, " ") + day, 0, 1)
        graph_ymd = year_only + pynini.cross(dash, " ") + month_number + day_optional

        final_graph = graph_dmy
        final_graph |= year_only
        final_graph |= graph_ymd

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
