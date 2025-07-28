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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.sv.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.sv.utils import get_abs_path as get_tn_abs_path
from nemo_text_processing.text_normalization.sv.utils import load_labels

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
        e.g. klockan åtta e s t -> time { hours: "kl. 8" zone: "e s t" }
        e.g. klockan tretton -> time { hours: "kl. 13" }
        e.g. klockan tretton tio -> time { hours: "kl. 13" minutes: "10" }
        e.g. kvart i tolv -> time { minutes: "45" hours: "11" }
        e.g. kvart över tolv -> time { minutes: "15" hours: "12" }

    Args:
        tn_cardinal_tagger: TN cardinal verbalizer
    """

    def __init__(self, tn_cardinal_tagger: GraphFst):
        super().__init__(name="time", kind="classify")

        suffixes = pynini.invert(pynini.string_map(load_labels(get_abs_path("data/time/suffix.tsv"))))
        self.suffixes = suffixes

        klockan = pynini.union(pynini.cross("klockan", "kl."), pynini.cross("klockan är", "kl."))
        klockan_graph_piece = pynutil.insert("hours: \"") + klockan
        minutes_to = pynini.string_map([(str(i), str(60 - i)) for i in range(1, 60)])
        minutes = pynini.string_map([str(i) for i in range(1, 60)])
        minutes_inverse = pynini.invert(pynini.project(minutes_to, "input") @ tn_cardinal_tagger.graph_en)
        minutes = pynini.invert(pynini.project(minutes, "input") @ tn_cardinal_tagger.graph_en)
        self.minute_words_to_words = minutes_inverse @ minutes_to @ tn_cardinal_tagger.graph_en
        self.minute_words_to_words_graph = (
            pynutil.insert("minutes: \"") + self.minute_words_to_words + pynutil.insert("\"")
        )

        time_zone_graph = pynini.invert(pynini.string_file(get_tn_abs_path("data/time/time_zone.tsv")))
        final_suffix = pynutil.insert("suffix: \"") + suffixes + pynutil.insert("\"")
        final_suffix_optional = pynini.closure(NEMO_SPACE + final_suffix, 0, 1)
        final_time_zone = pynutil.insert("zone: \"") + time_zone_graph + pynutil.insert("\"")
        final_time_zone_optional = pynini.closure(NEMO_SPACE + final_time_zone, 0, 1)
        both_optional_suffixes = final_suffix_optional + final_time_zone_optional
        one_optional_suffix = NEMO_SPACE + final_suffix + final_time_zone_optional
        one_optional_suffix |= final_suffix_optional + NEMO_SPACE + final_time_zone

        labels_hour = [str(x) for x in range(0, 24)]
        hours = pynini.invert(pynini.union(*labels_hour) @ tn_cardinal_tagger.graph)
        self.hours = hours
        hours_graph = pynutil.insert("hours: \"") + hours + pynutil.insert("\"")
        klockan_hour = klockan_graph_piece + NEMO_SPACE + hours + pynutil.insert("\"")
        hours_graph |= klockan_hour

        hour_sfx = hours_graph + one_optional_suffix

        def hours_to_pairs():
            for x in range(1, 13):
                if x == 12:
                    y = 1
                else:
                    y = x + 1
                yield x, y

        hours_to = pynini.string_map([(str(x[0]), str(x[1])) for x in hours_to_pairs()])
        hours_to = pynini.invert(hours_to @ tn_cardinal_tagger.graph)
        self.hours_to = hours_to
        hours_to_graph = pynutil.insert("hours: \"") + hours_to + pynutil.insert("\"")

        bare_quarters_to = pynini.string_map([(x[1], str(x[0])) for x in QUARTERS.items() if not "över" in x[1]])
        bare_quarters_from = pynini.cross("kvart över", "15")
        self.quarters_to = bare_quarters_to
        self.quarters_from = bare_quarters_from
        prefix_minutes_to = bare_quarters_to
        prefix_minutes_from = bare_quarters_from

        from_to_output = get_all_to_or_from_fst(tn_cardinal_tagger)

        for _, word in QUARTERS.items():
            for when in ["över", "i"]:
                num_part = pynini.invert(from_to_output[word][when])
                num_part_end = num_part + pynutil.delete(f" {when} {word}")
                if word == "kvart över":
                    prefix_minutes_from |= num_part_end
                else:
                    prefix_minutes_to |= num_part_end
        prefix_minutes_to |= minutes_inverse + pynutil.delete(" i")
        prefix_minutes_from |= minutes + pynutil.delete(" över")
        prefix_minutes_to_graph = pynutil.insert("minutes: \"") + prefix_minutes_to + pynutil.insert("\"")
        graph_to_prefixed = prefix_minutes_to_graph + NEMO_SPACE + hours_to_graph
        prefix_minutes_from_graph = pynutil.insert("minutes: \"") + prefix_minutes_from + pynutil.insert("\"")
        graph_from_prefixed = prefix_minutes_from_graph + NEMO_SPACE + hours_graph
        minutes_graph = pynutil.insert("minutes: \"") + minutes + pynutil.insert("\"")
        seconds_graph = pynutil.insert("seconds: \"") + minutes + pynutil.insert("\"")

        hm_sfx = hours_graph + NEMO_SPACE + minutes_graph + one_optional_suffix
        hms_sfx = hours_graph + NEMO_SPACE + minutes_graph + NEMO_SPACE + seconds_graph + one_optional_suffix

        graph = graph_to_prefixed | graph_from_prefixed | klockan_hour + both_optional_suffixes | hour_sfx
        graph |= hm_sfx
        graph |= hms_sfx
        self.fst = self.add_tokens(graph).optimize()
