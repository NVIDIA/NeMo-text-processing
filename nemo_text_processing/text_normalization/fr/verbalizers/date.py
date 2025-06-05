# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date {day: "deux" month: "mars" year: "deux mille trois" preserve_order: true} -> deux mars deux mille trois
    Args:
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, project_input: bool = False):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic, project_input=project_input)

        day = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        year = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        decade = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        graph_dmy = day + NEMO_SPACE + month + pynini.closure(NEMO_SPACE + year, 0, 1) + delete_preserve_order
        graph_my = month + NEMO_SPACE + year + delete_preserve_order
        graph_decade = decade + delete_preserve_order

        self.graph = graph_dmy | graph_my | graph_decade

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
