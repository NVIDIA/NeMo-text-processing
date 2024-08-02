# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, apply_fst
from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    insert_space,
    delete_space,
) 
from pynini.lib import pynutil, rewrite
class DateFst(GraphFst):
    """
    Finite state transducer for classifying fraction
          Finite state transducer for classifying date, 
        e.g. पांच जनवरी दो हज़ार बारह -> date { month: "जनवरी" day: "५" year: "२०१२" preserve_order: true }
        e.g. दो हज़ार बारह -> date { year: "२०१२" preserve_order: true }     
    Args:
        cardinal: CardinalFst
        date: DateFst
    """
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")
        
        cardinal_graph = cardinal.graph_no_exception
        
        
        month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_date_days = pynini.string_file(get_abs_path("data/date/date_days.tsv")).invert()
        graph_date = graph_digit | graph_date_days
        self.graph_date = graph_date
        
        insert_comma = pynutil.insert(",")

        
        self.day = pynini.closure(pynutil.insert("day: \"") + graph_date + pynutil.insert("\" "))
        self.month = pynini.closure(pynutil.insert("month: \"") + month_graph + pynutil.insert("\" "))
        self.year = pynini.closure(pynutil.insert("year: \"") + cardinal_graph + pynutil.insert("\" "))
        insert_comma = pynutil.insert(", ") 
        
        graph_date = self.day + delete_space + self.month + pynini.closure(delete_space + self.year, 0,1)

        graph = graph_date 
        self.graph = graph.optimize()
        
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
        
#from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
#cardinal = CardinalFst()
#date = DateFst(cardinal)
#input_text = "पच्चीस मार्च दो हज़ार दस"
#input_text = "तीन फ़रवरी"
#output = apply_fst(input_text, date.fst)
#print(output) 