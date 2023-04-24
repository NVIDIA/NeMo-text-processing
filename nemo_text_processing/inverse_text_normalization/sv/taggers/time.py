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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from pynini.lib import pynutil

QUARTERS = {15: "kvart över", 30: "halv", 45: "kvart i"}


def get_all_to_or_from_numbers():
    output = {}
    for num, word in QUARTERS.items():
        current_past = []
        current_to = []
        for i in range(1, 60):
            if i == num:
                continue
            elif i < num:
                current_to.append((str(i), str(num - i)))
            else:
                current_past.append((str(i), str(i - num)))
        output[word] = {}
        output[word]["över"] = current_past
        output[word]["i"] = current_to
    return output


def get_all_to_or_from_fst(cardinal: GraphFst):
    numbers = get_all_to_or_from_numbers()
    output = {}
    for key in numbers:
        output[key] = {}
        for when in ["över", "i"]:
            map = pynini.string_map(numbers[key][when])
            output[key][when] = pynini.project(map, "input") @ map @ cardinal.graph
    return output


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. klockan åtta e s t -> time { hours: "8" zone: "e s t" }
        e.g. klockan tretton -> time { hours: "13" }
        e.g. klockan tretton tio -> time { hours: "13" minutes: "10" }
        e.g. kvart i tolv -> time { minutes: "45" hours: "11" }
        e.g. kvart över tolv -> time { minutes: "15" hours: "12" }
        e.g. halv tolv -> time { minutes: "30" hours: "11" }
        e.g. tre i tolv -> time { minutes: "57" hours: "11" }
        e.g. tre i kvart i tolv -> time { minutes: "42" hours: "11" }
        e.g. tre över kvart i tolv -> time { minutes: "48" hours: "11" }
        e.g. tre över tolv -> time { minutes: "3" hours: "12" }
    
    Args:
        tn_time_verbalizer: TN time verbalizer
    """

    def __init__(self, tn_cardinal_tagger: GraphFst, tn_time_verbalizer: GraphFst):
        super().__init__(name="time", kind="classify")

        klockan = pynini.union(pynini.cross("klockan", "kl."), pynini.cross("klockan är", "kl."))
        klockan_graph_piece = pynutil.insert("hours: \"") + klockan
        minutes_to = pynini.string_map([(str(i), str(60 - i)) for i in range(1, 60)])
        minutes_inverse = pynini.invert(pynini.project(minutes_to, "input") @ tn_cardinal_tagger.graph_en)
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
        hours_to_graph = pynutil.insert(" hours: \"") + hours_to + pynutil.insert("\"")
        bare_quarters = pynini.string_map([(x[1], str(x[0])) for x in QUARTERS.items()])
        bare_quarters_graph = pynutil.insert("minutes: \"") + bare_quarters + pynutil.insert("\"")
        prefix_minutes = bare_quarters

        from_to_output = get_all_to_or_from_fst(tn_cardinal_tagger)

        for _, word in QUARTERS.items():
            for when in ["över", "i"]:
                num_part = pynini.invert(from_to_output[word][when])
                prefix_minutes |= num_part + pynutil.insert(f" {when} {word}")
        prefix_minutes_graph = pynutil.insert("minutes: \"") + prefix_minutes + pynutil.insert("\"")
        graph_prefixed = prefix_minutes_graph + hours_to_graph

        graph = graph_prefixed
        self.fst = self.add_tokens(graph).optimize()
