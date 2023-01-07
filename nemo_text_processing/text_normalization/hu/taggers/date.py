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
from nemo_text_processing.text_normalization.hu.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    TO_LOWER,
    GraphFst,
    insert_space,
)
from pynini.lib import pynutil


def get_suffixed_days(labels):
    endings = ["je", "a", "e"]
    output = []
    for label in labels:
        for ending in endings:
            if label[1].endswith(ending):
                output.append((f"{label[0]}-{ending}", label[1]))
                break
    return output


def day_inflector(number, day):
    """
    Generates pairs of inflected day numbers and their full forms,
    according to the options listed here:
    https://helyesiras.mta.hu/helyesiras/default/akh#298

    Args:
        number: the day number
        day: the day name
    
    Returns:
        a list of expanded forms, two per ending.
    """
    endings = {
        "e": "ét ének ével éért évé éig eként éül ében én énél ébe ére éhez éből éről étől",
        "a": "át ának ával áért ává áig aként ául ában án ánál ába ára ához ából áról ától",
    }
    output = []
    daylast = day[-1]
    for ending in endings[daylast].split(" "):
        daybase = day[:-1]
        endtrimmed = ending[1:]
        if day.endswith("eje"):
            output.append((f"{number}-j{ending}", f"{daybase}{ending}"))
            output.append((f"{number}-{ending}", f"{daybase}{ending}"))
        else:
            output.append((f"{number}-{ending}", f"{daybase}{ending}"))
            output.append((f"{number}-{endtrimmed}", f"{daybase}{ending}"))
    return output


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g. 
        "01.04.2010" -> date { day: "erster" month: "april" year: "zwei tausend zehn" preserve_order: true }
        "1994" -> date { year: "neunzehn vier und neuzig" }
        "1900" -> date { year: "neunzehn hundert" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        day_tsv = load_labels(get_abs_path("data/dates/days.tsv"))
        days_suffixed = get_suffixed_days(day_tsv)
        delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT

        month_abbr_graph = load_labels(get_abs_path("data/months/abbr_to_name.tsv"))
        number_to_month = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        month_graph = pynini.union(*[x[1] for x in month_abbr_graph]).optimize()
        month_abbr_graph = pynini.string_map(month_abbr_graph)
        month_abbr_graph = (
            pynutil.add_weight(month_abbr_graph, weight=0.0001)
            | ((TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete(".", weight=-0.0001), 0, 1)

        self.month_abbr = month_abbr_graph
        month_graph |= (TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_graph
        # jan.-> januar, Jan-> januar, januar-> januar
        month_graph |= month_abbr_graph

        numbers = cardinal.graph_hundred_component_at_least_one_none_zero_digit
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ numbers
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

        separators = ["."]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, " ") + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, " ") + month_number + year_optional
            graph_dmy |= new_graph

        dash = "-"
        day_optional = pynini.closure(pynini.cross(dash, " ") + day, 0, 1)
        graph_ymd = year_only + pynini.cross(dash, " ") + month_number + day_optional

        final_graph = graph_dmy + pynutil.insert(" preserve_order: true")
        final_graph |= year_only
        final_graph |= graph_ymd

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
