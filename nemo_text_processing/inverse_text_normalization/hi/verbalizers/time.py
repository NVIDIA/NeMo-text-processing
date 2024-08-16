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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "12" minutes: "30" } -> 12:30
        time { hours: "1" minutes: "12" } -> 01:12
        time { hours: "2" suffix: "a.m." } -> 02:00 a.m.
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("बजके"|"बजकर"|"बजे")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("मिनट")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete("सेकंड")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )

        #hour
        graph_hour = hour + delete_extra_space
        
        #hour minute second
        graph_hms = (
            hour + delete_extra_space + pynutil.insert(":") + delete_extra_space + minute + delete_extra_space + pynutil.insert(":") + delete_extra_space + second + delete_extra_space
        )
        
        #hour minute
        graph_hm = (
            hour + delete_extra_space + pynutil.insert(":") + delete_extra_space + minute + delete_extra_space
        )
        
        #hour second
        graph_hs = (
            hour + delete_extra_space + pynutil.insert(":") + delete_extra_space + second + delete_extra_space
        )
        
        #minute second
        graph_ms = (
            minute + delete_extra_space + pynutil.insert(":") + delete_extra_space + second + delete_extra_space
        )

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )

        final_graph = (graph_hour | graph_hms | graph_hm | graph_hs | graph_ms) + delete_extra_space + optional_preserve_order


        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()

        
#from nemo_text_processing.inverse_text_normalization.hi.taggers.time import TimeFst
#time = TimeFst()
#input_text = 'time { hour: "७" }'
#input_text = 'time { hour: "१२" minute: "०५"  }'
#input_text = 'time { hour: "७" second: "१२"  }'
#output = apply_fst(input_text, time.fst)
#print(output)