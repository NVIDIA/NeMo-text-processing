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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date e.g.
    date { year: "二千二十四" month: "三" day: "四" } -> 二千二十四年三月四日
  
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        year_component = (
            pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"") + pynutil.insert("年")
        )
        month_component = (
            pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"") + pynutil.insert("月")
        )
        day_component = (
            pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"") + pynutil.insert("日")
        )
        week_component = pynutil.delete("weekday: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_basic_date = (
            year_component
            + pynutil.delete(" ")
            + month_component
            + pynutil.delete(" ")
            + day_component
            + pynini.closure(pynutil.delete(" ") + week_component, 0, 1)
        )

        final_graph = graph_basic_date

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
