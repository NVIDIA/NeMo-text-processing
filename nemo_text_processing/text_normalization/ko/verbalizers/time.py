# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space, delete_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time e.g.


    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)
        
        hour_component = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        minute_content = pynini.closure(NEMO_NOT_QUOTE)
        minute_component = pynutil.delete("minutes: \"") + pynini.cross("영분", "") + pynutil.delete("\"") | \
                           pynutil.delete("minutes: \"") + (minute_content - "영분") + pynutil.delete("\"")

        second_content = pynini.closure(NEMO_NOT_QUOTE)
        second_component = pynutil.delete("seconds: \"") + pynini.cross("영초", "") + pynutil.delete("\"") | \
                           pynutil.delete("seconds: \"") + (second_content - "영초") + pynutil.delete("\"")

        division_component = pynutil.delete("suffix: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        
        graph_basic_time = pynini.closure(division_component + delete_space + insert_space, 0, 1) + (
            (hour_component + delete_space + insert_space + minute_component + delete_space + insert_space + second_component)
            | (hour_component + delete_space + insert_space + minute_component)
            | hour_component
            | minute_component
            | second_component
        )

        final_graph = graph_basic_time

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()