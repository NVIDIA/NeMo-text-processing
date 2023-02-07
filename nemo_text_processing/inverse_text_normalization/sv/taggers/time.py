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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. acht uhr e s t-> time { hours: "8" zone: "e s t" }
        e.g. dreizehn uhr -> time { hours: "13" }
        e.g. dreizehn uhr zehn -> time { hours: "13" minutes: "10" }
        e.g. viertel vor zwölf -> time { minutes: "45" hours: "11" }
        e.g. viertel nach zwölf -> time { minutes: "15" hours: "12" }
        e.g. halb zwölf -> time { minutes: "30" hours: "11" }
        e.g. drei vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei nach zwölf -> time { minutes: "3" hours: "12" }
        e.g. drei uhr zehn minuten zehn sekunden -> time { hours: "3" hours: "10" sekunden: "10"}
    
    Args:
        tn_time_verbalizer: TN time verbalizer
    """

    def __init__(self, tn_cardinal_tagger: GraphFst, tn_time_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        minutes_to = pynini.string_map([(str(i), str(60-i)) for i in range(1, 60)])
        minutes = pynini.string_map([str(i) for i in range(1, 60)])
        # minutes_to_words = minutes @ minutes_to @ tn_cardinal_tagger.graph_en
        minutes_inverse = pynini.invert(minutes @ tn_cardinal_tagger.graph_en)
        minute_words_to_words = minutes_inverse @ minutes_to @ tn_cardinal_tagger.graph_en
        minute_words_to_words = pynutil.insert("minutes: \"") + minute_words_to_words + pynutil.insert("\"")
        def hours_to_pairs():
            for x in range(1, 13):
                if x == 12:
                    y = 1
                else:
                    y = x + 1
                yield x, y
        hours_to = pynini.string_map([(str(x[0]), str(x[1])) for x in hours_to_pairs()])

        

        # lazy way to make sure compounds work
        optional_delete_space = pynini.closure(NEMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        graph = (tn_time_verbalizer.graph @ optional_delete_space).invert().optimize()
        self.fst = self.add_tokens(graph).optimize()
