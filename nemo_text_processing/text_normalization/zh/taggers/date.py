# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_CHAR, NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Date(GraphFst):
    '''
        2002年       -> tokens { date { year: "2002" } }
        2002-01-28   -> tokens { date { year: "2002" month: "01" day: "28"} }
        2002/01/28   -> tokens { date { year: "2002" month: "01" day: "28"} }
        2002.01.28   -> tokens { date { year: "2002" month: "01" day: "28"} }
        2002/02      -> tokens { date { year: "2002" month "02"} }
        02/11        -> tokens { date { year: "02" month "11"} } different with case "fraction 2/11"
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="classify", deterministic=deterministic)
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        year_whitelist = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))

        delete_date_sign = pynutil.delete("/") | pynutil.delete('-') | pynutil.delete('.')

        # 2012年
        date_type0 = (
            pynutil.insert("year: \"")
            + pynini.closure(graph_digit | graph_zero, 2, 4)
            + "年"
            + pynini.difference(NEMO_CHAR, year_whitelist)
            + pynutil.insert("\"")
        )

        year_2_4_digit = pynini.closure(NEMO_DIGIT, 2, 4) + delete_date_sign
        year_4_digit = pynini.closure(NEMO_DIGIT, 4, 4) + delete_date_sign
        year_2_digit_with_zero = "0" + NEMO_DIGIT + delete_date_sign
        month_no_day_with_zero = "0" + NEMO_DIGIT
        month_no_day = pynini.closure(NEMO_DIGIT, 2, 2)
        month = pynini.closure(NEMO_DIGIT, 1, 2) + delete_date_sign
        day = pynini.closure(NEMO_DIGIT, 1, 2)

        # 2012/01/28
        date_type1 = (
            pynutil.insert("year: \"")
            + year_2_4_digit
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("month: \"")
            + month
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("day: \"")
            + day
            + pynutil.insert("\"")
        )

        # 12/01
        date_type2 = (
            pynutil.insert("year: \"")
            + year_2_4_digit
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("month: \"")
            + month_no_day_with_zero
            + pynutil.insert("\"")
        )

        # 2012/11
        date_type3 = (
            pynutil.insert("year: \"")
            + year_4_digit
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("month: \"")
            + month_no_day
            + pynutil.insert("\"")
        )

        # 02/05
        date_type4 = (
            pynutil.insert("year: \"")
            + year_2_digit_with_zero
            + pynutil.insert("\"")
            + insert_space
            + pynutil.insert("month: \"")
            + month_no_day
            + pynutil.insert("\"")
        )
        # add your date type as date_typex here.
        graph = date_type0 | date_type1 | date_type2 | date_type3 | date_type4

        self.fst = self.add_tokens(graph).optimize()
