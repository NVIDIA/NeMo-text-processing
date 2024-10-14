# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
    15:30:30 tokens { time { hours: "15" minutes: "30" seconds: "30" preserve_order: true } } -> quindici e mezza trenta secondi
    12:15 tokens { time { hours: "12" minutes: "15" } } -> dodici e un quarto
    03:38 tokens { time { hours: "3" minutes: "38" } } -> tre e trentotto minuti

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        labels_hour = [str(x) for x in range(0, 25)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT

        graph_hour = pynini.union(*labels_hour)

        graph_minute_single = pynini.union(*labels_minute_single)
        graph_minute_double = pynini.union(*labels_minute_double)

        final_graph_hour_only = pynutil.insert("hours: \"") + graph_hour + pynutil.insert("\"")
        final_graph_hour = (
            pynutil.insert("hours: \"") + delete_leading_zero_to_double_digit @ graph_hour + pynutil.insert("\"")
        )
        final_graph_minute = (
            pynutil.insert("minutes: \"")
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        final_graph_second = (
            pynutil.insert("seconds: \"")
            + (pynutil.delete("0") + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )

        graph_hm = (
            final_graph_hour + pynutil.delete(":") + (pynutil.delete("00") | (insert_space + final_graph_minute))
        )

        graph_hms = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", " minutes: \"0\"") | (insert_space + final_graph_minute))
            + pynutil.delete(":")
            + (pynini.cross("00", " seconds: \"0\"") | (insert_space + final_graph_second))
            + pynutil.insert(" preserve_order: true")
        )

        graph_h = final_graph_hour_only

        final_graph = (graph_hm | graph_h | graph_hms).optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
