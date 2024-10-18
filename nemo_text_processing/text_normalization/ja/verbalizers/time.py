# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time e.g.
   
  
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour_component = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        minute_component = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        second_component = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        division_component = pynutil.delete("suffix: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_basic_time = pynini.closure(division_component + pynutil.delete(" "), 0, 1) + (
            (hour_component + pynutil.delete(" ") + minute_component + pynutil.delete(" ") + second_component)
            | (hour_component + pynutil.delete(" ") + minute_component)
            | hour_component
            | minute_component
            | second_component
        )

        final_graph = graph_basic_time

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
