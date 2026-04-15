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


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing Portuguese time, e.g.
        time { hours: "catorze" minutes: "trinta" preserve_order: true } -> catorze horas e trinta
        time { hours: "catorze" minutes: "trinta" seconds: "cinco" preserve_order: true }
        -> catorze horas e trinta minutos e cinco segundos
        time { hours: "onze" suffix: "da manhã" preserve_order: true } -> onze horas da manhã
        time { hours: "doze" preserve_order: true } -> doze horas

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        quoted = pynini.closure(NEMO_NOT_QUOTE, 1)

        hours = pynutil.delete('hours: "') + quoted + pynutil.delete('"')
        minutes_val = pynutil.delete('minutes: "') + quoted + pynutil.delete('"')
        seconds_val = pynutil.delete('seconds: "') + quoted + pynutil.delete('"')
        suffix_val = pynutil.delete('suffix: "') + quoted + pynutil.delete('"')

        gap = delete_space + insert_space
        suffix_out = pynini.closure(gap + suffix_val, 0, 1)

        graph_hms = (
            hours
            + gap
            + pynutil.insert("horas")
            + insert_space
            + pynutil.insert("e")
            + insert_space
            + minutes_val
            + gap
            + pynutil.insert("minutos")
            + insert_space
            + pynutil.insert("e")
            + insert_space
            + seconds_val
            + gap
            + pynutil.insert("segundos")
            + suffix_out
            + delete_preserve_order
        )

        with_minutes = (
            hours
            + gap
            + pynutil.insert("horas")
            + gap
            + pynutil.insert("e")
            + insert_space
            + gap
            + minutes_val
            + suffix_out
            + delete_preserve_order
        )

        hours_only = hours + gap + pynutil.insert("horas") + suffix_out + delete_preserve_order

        graph = pynini.union(graph_hms, with_minutes, hours_only).optimize()
        self.fst = self.delete_tokens(graph).optimize()
