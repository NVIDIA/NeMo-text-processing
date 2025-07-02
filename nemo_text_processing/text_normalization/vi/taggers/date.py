# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.utils import load_labels, get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst


class DateFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese dates, e.g.
        15/01/2024 -> date { day: "mười lăm" month: "một" year: "hai nghìn hai mươi tư" }
        tháng 4 2024 -> date { month: "tư" year: "hai nghìn hai mươi tư" }
        ngày 15/01/2024 -> date { day: "mười lăm" month: "một" year: "hai nghìn hai mươi tư" }
        ngày 12 tháng 5 năm 2025 -> date { day: "mười hai" month: "năm" year: "hai nghìn hai mươi lăm" }
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        day_mappings = load_labels(get_abs_path("data/date/days.tsv"))
        month_mappings = load_labels(get_abs_path("data/date/months.tsv"))
        
        day_digit = pynini.closure(NEMO_DIGIT, 1, 2)    
        month_digit = pynini.closure(NEMO_DIGIT, 1, 2)
        year_digit = pynini.closure(NEMO_DIGIT, 4, 4)  
        separator = pynini.union("/", "-", ".")
        
        day_convert = pynini.string_map([(k, v) for k, v in day_mappings]) | pynini.compose(day_digit, cardinal.graph)
        month_convert = pynini.string_map([(k, v) for k, v in month_mappings])
        year_convert = pynini.compose(year_digit, cardinal.graph)
        
        day_part = pynutil.insert("day: \"") + day_convert + pynutil.insert("\" ")
        month_part = pynutil.insert("month: \"") + month_convert + pynutil.insert("\" ")
        year_part = pynutil.insert("year: \"") + year_convert + pynutil.insert("\"")
        month_final = pynutil.insert("month: \"") + month_convert + pynutil.insert("\"")
        
        patterns = []
        
        date_sep = day_part + pynutil.delete(separator) + month_part + pynutil.delete(separator) + year_part
        patterns.append(pynini.compose(day_digit + separator + month_digit + separator + year_digit, date_sep))
        patterns.append(pynini.compose(
            pynini.accep("ngày ") + day_digit + separator + month_digit + separator + year_digit,
            pynutil.delete("ngày ") + date_sep
        ))
        
        for sep in [separator, pynini.accep(" ")]:
            patterns.append(pynini.compose(
                pynini.accep("tháng ") + month_digit + sep + year_digit,
                pynutil.delete("tháng ") + month_part + pynutil.delete(sep) + year_part
            ))
        
        day_month_sep = day_part + pynutil.delete(separator) + month_final
        patterns.append(pynini.compose(
            pynini.accep("ngày ") + day_digit + separator + month_digit,
            pynutil.delete("ngày ") + day_month_sep
        ))
        
        patterns.append(pynini.compose(
            pynini.accep("ngày ") + day_digit + pynini.accep(" tháng ") + month_digit,
            pynutil.delete("ngày ") + day_part + pynutil.delete(" tháng ") + month_final
        ))
        
        patterns.append(pynini.compose(
            pynini.accep("ngày ") + day_digit + pynini.accep(" tháng ") + month_digit + pynini.accep(" năm ") + year_digit,
            pynutil.delete("ngày ") + day_part + pynutil.delete(" tháng ") + month_part + pynutil.delete(" năm ") + year_part
        ))
        
        patterns.append(pynini.compose(
            pynini.accep("năm ") + year_digit,
            pynutil.delete("năm ") + year_part
        ))
        
        self.fst = self.add_tokens(pynini.union(*patterns))