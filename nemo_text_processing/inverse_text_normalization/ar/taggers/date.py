# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.ar.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying spoken Arabic date into a digit form,
    by inverting the text-normalization date tagger and verbalizer, e.g.
        الأول من شهر نوفمبر لعام ألفين وعشرة -> tokens { name: "01/11/2010" }
        الأول من شهر نوفمبر -> tokens { name: "1 نوفمبر" }
        الأول من شهر ربيع الثاني لعام ... هجري -> tokens { name: "01/04/1445 هـ" }

    Args:
        itn_cardinal_tagger: ITN cardinal tagger
        tn_date_tagger: TN date tagger
        tn_date_verbalizer: TN date verbalizer
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        tn_date_tagger: GraphFst,
        tn_date_verbalizer: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        optional_delete_space = pynini.closure(NEMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        ordinals = pynini.string_file(get_abs_path("data/ordinal/ordinals_date.tsv")).invert().optimize()
        hijri_suffixes = pynini.string_file(get_abs_path("data/months/hijri_suffixes_inverse.tsv")).optimize()
        months_names = [x[0] for x in load_labels(get_abs_path("data/months/months_name.tsv"))]
        months_names = pynini.union(*months_names)
        month_to_number = tn_date_tagger.number_to_month.invert().optimize()
        month_to_number_hijri = tn_date_tagger.month_hijri.invert().optimize()
        tagger = tn_date_verbalizer.graph.invert().optimize()

        delete_day_marker = (
            pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        ) @ ordinals

        month_as_number = pynutil.delete("month: \"") + itn_cardinal_tagger.graph + pynutil.delete("\"")
        month_as_string = pynutil.delete("month: \"") + months_names + pynutil.delete("\"")
        month_name_to_number = (
            pynutil.delete("month: \"") + (month_to_number | month_to_number_hijri) + pynutil.delete("\"")
        )

        convert_year = (tn_date_tagger.year @ optional_delete_space).invert().optimize()
        convert_year |= (
            (tn_date_tagger.year @ optional_delete_space).invert().optimize() + pynini.accep(" ") + hijri_suffixes
        )
        delete_year_marker = (
            pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        ) @ convert_year

        # day month as string (year)
        verbalizer = pynutil.add_weight(
            (
                pynini.closure(delete_day_marker + pynini.accep(" "), 0, 1)
                + month_as_string
                + pynini.closure(pynini.accep(" ") + delete_year_marker, 0, 1)
            ),
            weight=0.0001,
        )

        # day month as number (year); trailing "/" only when a year is present
        verbalizer |= (
            delete_day_marker @ add_leading_zero_to_double_digit
            + pynutil.insert("/")
            + pynutil.delete(" ")
            + month_as_number @ add_leading_zero_to_double_digit
            + pynini.closure(pynutil.insert("/") + pynutil.delete(" ") + delete_year_marker, 0, 1)
        )
        # 02/03/2022 هـ
        verbalizer |= (
            delete_day_marker @ add_leading_zero_to_double_digit
            + pynutil.insert("/")
            + pynutil.delete(" ")
            + month_name_to_number @ add_leading_zero_to_double_digit
            + pynini.closure(pynutil.insert("/") + pynutil.delete(" ") + delete_year_marker, 0, 1)
        )

        # year
        verbalizer |= delete_year_marker

        final_graph = tagger @ verbalizer

        graph = pynutil.insert("name: \"") + convert_space(final_graph) + pynutil.insert("\"")
        self.fst = graph.optimize()
