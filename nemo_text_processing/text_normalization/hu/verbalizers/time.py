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
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "tolv" minutes: "trettio" suffix: "förmiddag" zone: "e s t" } -> tolv trettio förmiddag e s t
        time { hours: "tolv" } -> tolv

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)
        ANY_NOT_QUOTE = pynini.closure(NEMO_NOT_QUOTE, 1)
        NOT_NULLA = pynini.difference(ANY_NOT_QUOTE, "nulla")
        hour = pynutil.delete("hours: \"") + ANY_NOT_QUOTE + pynutil.delete("\"")
        hour_ora = hour + pynutil.insert(" óra")
        minute = pynutil.delete("minutes: \"") + NOT_NULLA + pynutil.delete("\"")
        minute |= pynutil.delete("minutes: \"nulla\"")
        minute_perc = minute + pynutil.insert(" perc")
        suffix = pynutil.delete("suffix: \"") + ANY_NOT_QUOTE + pynutil.delete("\"")
        optional_suffix = pynini.closure(delete_space + insert_space + suffix, 0, 1)
        zone = (
            pynutil.delete("zone:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_zone = pynini.closure(delete_space + insert_space + zone, 0, 1)
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        second |= pynutil.delete("seconds: \"nulla\"")
        second_masodperc = second + pynutil.insert(" masodperc")
        graph = hour_ora + NEMO_SPACE + minute + NEMO_SPACE + suffix + optional_zone
        graph |= hour_ora + NEMO_SPACE + minute + NEMO_SPACE + suffix + delete_preserve_order
        graph |= (
            hour_ora + NEMO_SPACE + minute_perc + NEMO_SPACE + second + NEMO_SPACE + suffix + delete_preserve_order
        )
        graph |= hour_ora + NEMO_SPACE + second + NEMO_SPACE + suffix + delete_preserve_order
        graph |= (
            hour_ora + NEMO_SPACE + minute_perc + NEMO_SPACE + second_masodperc + optional_zone + delete_preserve_order
        )
        graph |= hour_ora + NEMO_SPACE + second_masodperc + optional_zone + delete_preserve_order
        graph |= hour + NEMO_SPACE + suffix + delete_preserve_order
        graph |= hour_ora + optional_zone + delete_preserve_order
        graph |= hour_ora + NEMO_SPACE + minute_perc + optional_zone + delete_preserve_order
        graph |= hour + NEMO_SPACE + minute + NEMO_SPACE + second + optional_suffix + optional_zone
        graph |= hour + NEMO_SPACE + suffix + optional_zone
        graph |= hour + optional_zone
        graph = (
            graph
            @ pynini.cdrewrite(delete_extra_space, "", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
        )
        # graph |= graph_hms
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
