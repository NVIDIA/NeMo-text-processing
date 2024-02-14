# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { day: "1." month: "jan." preserve_order: true } -> 1. jan.
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        year = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        day = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        era = pynutil.delete("era: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_era = pynini.closure(NEMO_SPACE + era, 0, 1)
        space_to_hyphen = pynini.cross(" ", "-")

        optional_preserve_order = pynini.closure(
            pynutil.delete(" preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete(" field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
        )

        # day month
        year_era = year + optional_era + optional_preserve_order
        graph_dm = day + NEMO_SPACE + month + delete_preserve_order
        graph_ydm = year + space_to_hyphen + month + space_to_hyphen + day + optional_era + optional_preserve_order

        final_graph = graph_dm | graph_ydm | year_era

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
