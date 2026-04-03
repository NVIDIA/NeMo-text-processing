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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing Portuguese (Brazilian) dates, e.g.
        date { day: "quinze" month: "março" year: "dois mil e vinte e quatro" preserve_order: true }
        -> quinze de março de dois mil e vinte e quatro
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        vrows = load_labels(get_abs_path("data/date/verbal_phrases.tsv"))
        vp = {r[0].strip(): r[1].strip() for r in vrows if len(r) >= 2 and r[0].strip()}
        prep = vp.get("preposition", "de") + " "

        quoted = pynini.closure(NEMO_NOT_QUOTE, 1)

        day_expr = pynutil.delete('day: "') + quoted + pynutil.delete('"')
        month_expr = pynutil.delete('month: "') + quoted + pynutil.delete('"')
        year_expr = pynutil.delete('year: "') + quoted + pynutil.delete('"')

        ws = delete_space + insert_space
        glue = ws + pynutil.insert(prep) + ws

        graph_dmy = day_expr + glue + month_expr + glue + year_expr + delete_preserve_order
        self.fst = self.delete_tokens(graph_dmy).optimize()
