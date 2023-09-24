# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from nemo_text_processing.inverse_text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path, num_to_word
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    capitalized_input_graph,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
      def __init__(self):
            super().__init__(name="time", kind="classify")
            hours = pynini.string_file(get_abs_path("data/time/hours.tsv"))
            minutes = pynini.string_map(get_abs_path("data/time/minutes.tsv"))
            quarter_times = pynini.string_map(get_abs_path("data/time/hours_to.tsv"))

            time_word = pynini.cross("वाजून","")
            minutes_word = pynini.cross("मिनिटे","") | pynini.cross("मिनिट","")
            graph_time_general = pynutil.insert("hours: \"") + hours + pynutil.insert("\"") + delete_space + time_word + delete_space + pynutil.insert(" ") + pynutil.insert("minutes: \"") + minutes + pynutil.insert("\"") + delete_space + minutes_word
            graph_time_to = pynutil.insert("hours: \"") + quarter_times + pynutil.insert("\"") + pynini.cross("ला","") + delete_space + pynutil.insert(" ") + pynutil.insert("minutes: \"") + minutes + pynutil.insert("\"") + delete_space + minutes_word
            # graph_oclock = pynutil.insert("hours: \"") + hours + pynutil.insert("\"") + delete_space + (pynini.cross("वाजले","")|pynini.cross("वाजता","")) + pynutil.insert(" ") + pynutil.insert("minutes: \"००\"")

            # graph_fifteen_thirty = pynutil.insert("minutes: \"") + terms + pynutil.insert("\"") + delete_space + pynutil.insert(" ") + pynutil.insert("hours: \"") + hours + pynutil.insert("\"")
            # graph_quarter_to = pynutil.insert("minutes: \"") + terms + pynutil.insert("\"") + delete_space + pynutil.insert(" ") + pynutil.insert("hours: \"") + quarter_times + pynutil.insert("\"")
            graph_fifteen = pynini.cross("सव्वा","") + delete_space + pynutil.insert("hours: \"") + hours + pynutil.insert("\"") + pynutil.insert(" ") + pynutil.insert("minutes: \"") + pynutil.insert("१५") + pynutil.insert("\"")
            graph_thirty = pynini.cross("साडे","") + delete_space + pynutil.insert("hours: \"") + hours + pynutil.insert("\"") + pynutil.insert(" ") + pynutil.insert("minutes: \"") + pynutil.insert("३०") + pynutil.insert("\"")
            graph_fortyfive = pynini.cross("पावणे","") + delete_space + pynutil.insert("hours: \"") + quarter_times + pynutil.insert("\"") + pynutil.insert(" ") + pynutil.insert("minutes: \"") + pynutil.insert("४५") + pynutil.insert("\"")

            special_cases = (pynini.cross("दीड","") + pynutil.insert("hours: \"१\" minutes: \"३०\"")) | (pynini.cross("अडीच","") + pynutil.insert("hours: \"२\" minutes: \"३०\""))

            # graph = pynini.union(graph_time_general, graph_time_to, graph_fifteen_thirty, graph_quarter_to, special_cases)
            graph = pynini.union(graph_time_general, graph_time_to, graph_fifteen, graph_thirty, graph_fortyfive, special_cases)

            # final_graph = pynutil.insert("time: \"") + graph + pynutil.insert("\"")
            final_graph = graph
            final_graph = self.add_tokens(final_graph)
            self.fst = final_graph.optimize()
            # वाजून बारा मिनिटे