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
from nemo_text_processing.inverse_text_normalization.fr.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "8" minutes: "30" suffix: "du matin"} -> 8 h 30
        time { hours: "8" minutes: "30" } -> 8 h 30
        time { hours: "8" minutes: "30" suffix: "du soir"} -> 20 h 30  
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour_to_night = pynini.string_file(get_abs_path("data/time/hour_to_night.tsv"))

        day_suffixes = pynutil.delete("suffix: \"am\"")
        night_suffixes = pynutil.delete("suffix: \"pm\"")

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_extra_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete("\"")
        )

        graph = hour + delete_extra_space + pynutil.insert("h") + minute.ques + delete_space + day_suffixes.ques

        graph |= (
            hour @ hour_to_night
            + delete_extra_space
            + pynutil.insert("h")
            + minute.ques
            + delete_space
            + night_suffixes
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
