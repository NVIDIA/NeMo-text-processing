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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_preserve_order
from nemo_text_processing.text_normalization.it.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
    tokens { time { hours: "15" minutes: "30" seconds: "30" preserve_order: true } } -> quindici e mezza trenta secondi
    tokens { time { hours: "12" minutes: "15" } } -> dodici e un quarto
    tokens { time { hours: "3" minutes: "38" } } -> tre e trentotto minuti
    Args:
        cardinal_tagger: cardinal_tagger tagger GraphFst
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        graph_zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv"))).optimize()
        number_verbalization = graph_zero | cardinal_tagger.two_digit_no_zero
        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        hour_verbalized = hour @ number_verbalization
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")

        graph_hms_15 = (
            hour_verbalized
            + pynini.accep(" ")
            + pynutil.insert("e ")
            + (minute @ number_verbalization + pynutil.insert(" minuti") | minute @ pynini.cross("15", "un quarto"))
            + pynini.accep(" ")
            + pynutil.insert("e ")
            + second @ number_verbalization
            + pynutil.insert(" secondi")
        )

        graph_hms_30 = (
            hour_verbalized
            + pynini.accep(" ")
            + pynutil.insert("e ")
            + (minute @ number_verbalization + pynutil.insert(" minuti") | minute @ pynini.cross("30", " mezza"))
            + pynini.accep(" ")
            + pynutil.insert("e ")
            + second @ number_verbalization
            + pynutil.insert(" secondi")
        )

        graph_hm_15 = (
            hour_verbalized
            + pynini.accep(" ")
            + pynutil.insert("e ")
            + (minute @ number_verbalization + pynutil.insert(" minuti") | minute @ pynini.cross("15", "un quarto"))
        )

        graph_hm_30 = (
            hour_verbalized
            + pynini.accep(" ")
            + pynutil.insert("e ")
            + (minute @ number_verbalization + pynutil.insert(" minuti") | minute @ pynini.cross("30", " mezza"))
        )

        graph_h = hour_verbalized

        self.graph = graph_hms_30 | graph_hms_30 | graph_hms_15 | graph_hm_30 | graph_hm_15 | graph_h

        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
