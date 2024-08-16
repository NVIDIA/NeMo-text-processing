# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, insert_space, NEMO_HI_DIGIT, NEMO_HI_NON_ZERO, NEMO_HI_ZERO
from nemo_text_processing.text_normalization.hi.utils import get_abs_path, apply_fst
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from pynini.lib import pynutil

days = pynini.string_file(get_abs_path("data/date/days.tsv"))
months = pynini.string_file(get_abs_path("data/date/months.tsv"))
digit = pynini.string_file(get_abs_path("data/date/digit.tsv"))
non_zero_digit = pynini.string_file(get_abs_path("data/date/non_zero_digit.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "०१-०४-२०२४" -> date { day: "एक" month: "अप्रैल" year: "दो हज़ार चौबीस" }
        "०४-०१-२०२४" -> date { month: "अप्रैल" day: "एक" year: "दो हज़ार चौबीस" }
        "२०२४-०१-०४" -> date { year: "दो हज़ार चौबीस" day: "एक" month: "अप्रैल" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")
        
        graph_year = pynini.compose((NEMO_HI_DIGIT + NEMO_HI_ZERO + NEMO_HI_DIGIT + NEMO_HI_DIGIT), cardinal.graph_thousands)
        graph_year_as = pynini.compose((NEMO_HI_DIGIT + NEMO_HI_NON_ZERO + NEMO_HI_DIGIT + NEMO_HI_DIGIT), cardinal.graph_hundred_as_thousand)
        graph_years = graph_year | graph_year_as
        
        delete_dash = pynutil.delete("-")
    
        days_graph = pynutil.insert("day: \"") + days + pynutil.insert("\"") + insert_space
        
        months_graph = pynutil.insert("month: \"") + months + pynutil.insert("\"") + insert_space
        
        years_graph = pynutil.insert("year: \"") + graph_years + pynutil.insert("\"") + insert_space
             
        graph_dd_mm_yyyy = days_graph + delete_dash + months_graph + delete_dash + years_graph
        
        graph_mm_dd_yyyy = months_graph + delete_dash + days_graph + delete_dash + years_graph
        
        graph_mm_dd = months_graph + delete_dash + days_graph 
        
        graph_dd_mm = days_graph + delete_dash +  months_graph
        
        graph_mm_yyyy = months_graph + delete_dash +  years_graph
        
        graph_yyyy_mm = years_graph + delete_dash + months_graph
        
        final_graph = graph_dd_mm_yyyy | graph_mm_dd_yyyy | graph_mm_dd | graph_dd_mm | graph_mm_yyyy | graph_yyyy_mm
        
        self.final_graph = final_graph.optimize()
        
        self.fst = self.add_tokens(self.final_graph)

        
cardinal = CardinalFst()
date = DateFst(cardinal)
#input_text = "१७-०१-१०१७" # १७ २५ ३१ ३७ ४४ ५० ५९ ७० ७९ ८९ ९९ 
input_text = "१७-०२-११००"  
#input_text = "१७-०३-१२००" 
#input_text = "२५-०३-१३९९"
#input_text = "३१-०९-१४००"
#input_text = "०१-०४-१५९९"
#input_text = "०४-०७-१६९९"
#input_text = "०९-०५-१७८९"
#input_text = "२९-०२-१८९९"  # १८०० १८०१ १८०९ १८१० १८३७ १८९९
#input_text = "२९-०२-१९९९"
#input_text = "१६-०६-२०९९"
#input_text = "१९-०९-२१००"
#input_text = "१९-०९-२२९९"
#input_text = "१९-०९-२३९९"
#input_text = "१९-०९-२४९९"
#input_text = "१९-०९-२५३१"
#input_text = "०९-१९-२५३१"
#input_text = "०२-१९"
#input_text = "१९-०९"
#input_text = "०९-२०२४"
#input_text = "२०२४-०९"
output = apply_fst(input_text, date.fst)  
#print(dir(cardinal))