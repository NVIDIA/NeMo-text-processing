# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_DIGIT, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.hu.graph_utils import TO_LOWER, TO_UPPER
from nemo_text_processing.text_normalization.hu.utils import get_abs_path, load_labels


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


def day_adj_endings(number, word, basic=True):
    """
    Two adjective forms can be formed from the days (three for 1):
        1-i -> elseji
        1-ji -> elseji
        1-jei -> elsejei
        2-i -> másodiki
        2-ai -> másodikai
        4-i -> negyediki
        4-ei -> negyedikei
    This is based on other -i adjectives, because these forms are rare.
    """
    endings_pl = {
        "e": "iek ieket ieknek iekkel iekért iekké iekig iekként iekben ieken ieknél iekbe iekre iekhez iekből iekről iektől",
        "a": "iak iakat iaknak iakkal iakért iakká iakig iakként iakban iakon iaknál iakba iakra iakhoz iakból iakról iaktól",
    }
    endings_sg = {
        "e": "i it inek ivel iért ivé iig iként iben in inél ibe ire ihez iből iről itől",
        "a": "i it inak ival iért ivá iig iként iban in inál iba ira ihoz iból iról itól",
    }
    last = word[-1]
    short = word[:-1]
    output = []
    if basic:
        endings = ["i"]
    else:
        endings = endings_sg[last].split(" ") + endings_pl[last].split(" ")
    for ending in endings:
        if word == "elseje":
            output.append((f"{number}-{ending}", f"{short}{ending}"))
            output.append((f"{number}-j{ending}", f"{short}{ending}"))
            output.append((f"{number}-{last}{ending}", f"{word}{ending}"))
        else:
            output.append((f"{number}-{ending}", f"{short}{ending}"))
            output.append((f"{number}-{last}{ending}", f"{word}{ending}"))
    return output


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "2010. április 1." -> date { year: "kettőezer-tíz" month: "április" day: "elseje" preserve_order: true }
        "2010. ápr. 1." -> date { year: "kettőezer-tíz" month: "április" day: "elseje" preserve_order: true }
        "2010. IV. 1." -> date { year: "kettőezer-tíz" month: "április" day: "elseje" preserve_order: true }
        "2010. 04. 1." -> date { year: "kettőezer-tíz" month: "április" day: "elseje" preserve_order: true }
        "2010. 04. 1-je" -> date { year: "kettőezer-tíz" month: "április" day: "elseje" preserve_order: true }
        "2010. 04. 1-jén" -> date { year: "kettőezer-tíz" month: "április" day: "elsején" preserve_order: true }
        "2010. 04. 1-én" -> date { year: "kettőezer-tíz" month: "április" day: "elsején" preserve_order: true }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)
        delete_dot = pynutil.delete(".")
        optional_dot = pynini.closure(delete_dot, 0, 1)

        day_tsv = load_labels(get_abs_path("data/dates/days.tsv"))
        graph_day = pynini.string_map(day_tsv)
        days_suffixed = get_suffixed_days(day_tsv)

        use_full_adj_forms = False
        if not deterministic:
            use_full_adj_forms = True
        day_adj_forms = []
        for day in day_tsv:
            days_suffixed += day_inflector(day[0], day[1])
            day_adj_forms += day_adj_endings(day[0], day[1], use_full_adj_forms)

        graph_days_suffixed = pynini.string_map(days_suffixed)
        graph_days_adj_suffixed = pynini.string_map(day_adj_forms)
        graph_days_suffixed |= pynini.project(graph_days_suffixed, "output")
        graph_days_adj_suffixed |= pynini.project(graph_days_adj_suffixed, "output")
        self.days_suffixed = graph_days_suffixed
        self.days_suffixed |= graph_days_adj_suffixed
        self.days_only = pynutil.insert("day: \"") + graph_days_suffixed + pynutil.insert("\"")

        # these express from and to, respectively
        # december 25-től január 27-ig -> from December 25 to January 27
        self.days_tol = (pynini.closure(NEMO_CHAR) + pynini.union("től", "tól")) @ graph_days_suffixed
        self.days_ig = (pynini.closure(NEMO_CHAR) + "ig") @ graph_days_suffixed

        delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT

        month_abbr_graph = load_labels(get_abs_path("data/dates/month_abbr.tsv"))
        number_to_month = pynini.string_file(get_abs_path("data/dates/months.tsv")).optimize()
        month_romans = pynini.string_file(get_abs_path("data/dates/months_roman.tsv")).optimize()
        month_romans |= pynini.invert(pynini.invert(month_romans) @ pynini.closure(TO_UPPER))
        month_romans_dot = month_romans + delete_dot
        month_graph = pynini.union(*[x[1] for x in month_abbr_graph]).optimize()
        month_abbr_graph = pynini.string_map(month_abbr_graph)

        self.month_abbr = month_abbr_graph
        month_graph |= (TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_graph
        # jan.-> januar, Jan-> januar, januar-> januar
        month_abbr_dot = month_abbr_graph + delete_dot

        numbers = cardinal.graph
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ graph_day
        day = (pynutil.insert("day: \"") + digit_day + pynutil.insert("\"")).optimize()
        day_dot = (pynutil.insert("day: \"") + digit_day + pynutil.delete(".") + pynutil.insert("\"")).optimize()
        self.day_dot = day_dot
        day_words = pynini.project(digit_day, "output")
        day_pieces = (digit_day + optional_dot) | day_words | graph_days_suffixed
        day_part = (pynutil.insert("day: \"") + day_pieces + pynutil.insert("\"")).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month
        number_to_month_dot = number_to_month + delete_dot
        month_part = month_abbr_dot | month_graph | month_romans_dot | number_to_month_dot
        self.month = month_part

        month_component = (pynutil.insert("month: \"") + month_part + pynutil.insert("\"")).optimize()
        month_number_only = (pynutil.insert("month: \"") + number_to_month + pynutil.insert("\"")).optimize()
        self.month_component = month_component
        self.month_number_only = month_number_only
        self.month_number = number_to_month_dot
        month_component = self.month_component.optimize()

        # prefer cardinal over year
        year = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 1, 3)  # 90, 990, 1990
        year @= numbers
        self.year = year

        year_only = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        year_dot = pynutil.insert("year: \"") + year + pynutil.delete(".") + pynutil.insert("\"")
        optional_year_dot_space = pynini.closure(year_dot + NEMO_SPACE, 0, 1)

        graph_ymd = optional_year_dot_space + month_component + NEMO_SPACE + day_part
        graph_ymd |= (
            pynini.closure(year_only + pynini.cross("-", " "), 0, 1) + month_number_only + pynini.cross("-", " ") + day
        )
        self.ymd = graph_ymd
        graph_ym = year_dot + NEMO_SPACE + month_component

        graph_dmy = (
            day + pynini.cross("-", " ") + month_number_only + pynini.closure(pynini.cross("-", " ") + year_only, 0, 1)
        )
        separators = [".", "/"]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, " ") + year_only, 0, 1)
            if not deterministic:
                new_graph = day + pynini.cross(sep, " ") + month_number_only + year_optional
            else:
                new_graph = day + pynini.cross(sep, " ") + month_number_only + year_only
            graph_dmy |= new_graph

        final_graph = graph_ymd + pynutil.insert(" preserve_order: true")
        final_graph |= graph_ym + pynutil.insert(" preserve_order: true")
        final_graph |= graph_dmy

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
