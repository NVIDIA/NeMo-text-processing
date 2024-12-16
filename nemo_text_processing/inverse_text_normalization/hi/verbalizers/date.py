# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)
 
from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst
 
 
class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "जनवरी" day: "५" year: "२०१२" preserve_order: true } -> जनवरी ५ २०१२
        date { day: "५" month: "जनवरी" year: "२०१२" preserve_order: true } -> ५ जनवरी २०१२
    """
 
    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )
        period = (
            pynutil.delete("text:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph_fy = year
        graph_fy |= period + delete_space + year
        
        #century
        graph_century = year + delete_extra_space + period
        
        # month (day) year
        graph_mdy = month + delete_extra_space + day + pynutil.insert(",") + delete_extra_space + year
 
        # (day) month year
        graph_dmy = day + delete_extra_space + month + pynutil.insert(",") + delete_extra_space + year
        
        # day month year century
        graph_dmyc = day + delete_extra_space + month + pynutil.insert(",") + delete_extra_space + year + delete_extra_space + period
 
        # month year
        graph_my = month + pynini.closure(delete_extra_space + year, 0, 1)
        
        # month year century
        graph_myc = month + pynutil.insert(",") + delete_extra_space + year + delete_extra_space + period
 
        # month day
        graph_md = month + pynini.closure(delete_extra_space + day, 0, 1)
 
        # day month
        graph_dm = day + pynini.closure(delete_extra_space + month, 0, 1)
        
        # year range
        graph_year_range = year
        
        # date exceptions
        #graph_date_exceptions = day + delete_extra_space + pynutil.insert("की") + delete_extra_space + month
 
        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )
 
        final_graph = (
            (graph_fy | graph_mdy | graph_dmy | graph_my | graph_md | graph_dm | graph_century | graph_dmyc | graph_myc | graph_year_range)
            + delete_space
            + optional_preserve_order
        )
 
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
        
#date = DateFst()
#input_text = 'date { period: "सन " year: "२०१९"  }'
#input_text = 'date { day: "१७"month: "अप्रैल"year: "२००२" }'
#input_text = 'date { day: "२५" month: "मार्च" year: "२०१०"  }'
#input_text = 'date { day: "१७" month: "अक्टूबर" year: "२०१९"  }'
#input_text = 'date { year: "४४००"  }'
#input_text = 'date { year: "४४००" text: "ईस्वी" }'
#input_text = 'date { year: "४४००" text: "ई. पू."  }'
#input_text = 'date { day: "२५" month: "मार्च" year: "२०१०" text: "ई. पू." }'
#input_text = 'date { year: "१९२०-२६" }'
#input_text = 'date { month: "फ़रवरी" day: "२०" }'
#output = apply_fst(input_text, date.fst)
#print(output)
