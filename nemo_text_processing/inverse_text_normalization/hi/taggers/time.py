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
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    insert_space,
    delete_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
        Finite state transducer for classifying time, 
        e.g. एक बजके सात मिनट -> time { hours: "१" minutes: "७" }
        e.g. चार बजे चवालीस मिनट -> time { hours: "४" minutes: "४४" }     
    Args:
        cardinal: CardinalFst
        time: TimeFst
    """
    def __init__(self):
        super().__init__(name="time", kind="classify")

        hour_graph = pynini.string_file(get_abs_path("data/time/hour.tsv")).invert()
        minute_graph = pynini.string_file(get_abs_path("data/time/minute.tsv")).invert()

        delete_baje = pynini.union(
            pynutil.delete("बजके") 
            | pynutil.delete("बजकर") 
            | pynutil.delete("बजे")
        )
        
        delete_minute = pynutil.delete("मिनट")
        
        self.hour = pynutil.insert("hour: \"") + hour_graph + pynutil.insert("\" ")
        self.minute = pynutil.insert("minute: \"") + minute_graph + pynutil.insert("\" ")

        graph_time = self.hour + pynini.closure(delete_space + delete_baje, 0,1) + self.minute + pynini.closure(delete_space + delete_minute, 0,1)

        graph = graph_time 
        self.graph = graph
        
        final_graph = self.add_tokens(graph)
        self.fst = final_graph