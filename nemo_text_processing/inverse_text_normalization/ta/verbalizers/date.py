# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing dates, e.g.
        date { day: "15" month: "ஜூன்" year: "2024" preserve_order: true } -> 15 ஜூன் 2024
        date { year: "2024" month: "ஜூன்" day: "15" preserve_order: true } -> 2024 ஜூன் 15
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        def field(name: str) -> 'pynini.FstLike':
            return pynutil.delete(f"{name}: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        day, month, year = field("day"), field("month"), field("year")
        sep = delete_space + insert_space
        graph = (
            day + sep + month + pynini.closure(sep + year, 0, 1)
            | month + sep + year
            | year + sep + month + sep + day
            | month + sep + day + pynini.closure(sep + year, 0, 1)
        )
        self.graph = graph + delete_preserve_order
        self.fst = self.delete_tokens(self.graph).optimize()
