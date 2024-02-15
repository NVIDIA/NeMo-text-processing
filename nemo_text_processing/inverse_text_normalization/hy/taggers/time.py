# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hy.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_space


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. տասներկուսն անց հինգ -> time { hours: "12" minutes: "05" }
        e.g. հինգին տասնհինգ պակաս -> time { hours: "04" minutes: "45" }
        e.g. տասներեք անց կես -> time { hours: "12" minutes: "30" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        graph_oclock = pynutil.delete("անց")

        graph_demi = pynini.cross("կես", "30")

        graph_fractions = graph_demi

        graph_hours = pynini.string_file(get_abs_path("data/time/hours.tsv")) + (
            pynini.closure(pynutil.delete("ն") | pynutil.delete("ին"), 0, 1)
        )
        graph_minutes = pynini.string_file(get_abs_path("data/time/minutes.tsv")) + (
            pynini.closure(pynutil.delete("ն") | pynutil.delete("ին"), 0, 1)
        )
        graph_hours_to = pynini.string_file(get_abs_path("data/time/to_hour.tsv"))
        graph_minutes_to = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))
        graph_to = pynutil.delete("պակաս")

        graph_hours_component = pynutil.insert("hours: \"") + graph_hours + pynutil.insert("\"")

        graph_minutes_component = (
            pynutil.insert(" minutes: \"") + pynini.union(graph_minutes, graph_fractions) + pynutil.insert("\"")
        )
        graph_minutes_component = delete_space + graph_minutes_component

        graph_time_standard = (
            graph_hours_component + delete_space + graph_oclock + pynini.closure(graph_minutes_component, 0, 1)
        )

        graph_hours_to_component = graph_hours + pynutil.delete('ին')
        graph_hours_to_component @= graph_hours_to
        graph_hours_to_component = pynutil.insert("hours: \"") + graph_hours_to_component + pynutil.insert("\"")

        graph_minutes_to_component = graph_minutes
        graph_minutes_to_component @= graph_minutes_to
        graph_minutes_to_component = pynutil.insert(" minutes: \"") + graph_minutes_to_component + pynutil.insert("\"")

        graph_time_to = graph_hours_to_component + delete_space + graph_minutes_to_component + delete_space + graph_to

        graph_time_no_suffix = graph_time_standard | graph_time_to

        final_graph = graph_time_no_suffix

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
